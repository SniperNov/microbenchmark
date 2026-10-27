#ifndef MICROBENCHMARK_HIP_OPENMP_PROFILE_H
#define MICROBENCHMARK_HIP_OPENMP_PROFILE_H

// Host-only reader for one-repetition OpenMP delay observations. This format
// records owners and counts, not timestamps or parallel-region boundaries.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace microbenchmark_openmp_profile {

struct Key {
    int method_id, N, max_iter, max_array_size;
    int requested_team_count, requested_thread_count;
    bool operator<(const Key& other) const {
        return std::tie(method_id, N, max_iter, max_array_size,
                        requested_team_count, requested_thread_count) <
               std::tie(other.method_id, other.N, other.max_iter, other.max_array_size,
                        other.requested_team_count, other.requested_thread_count);
    }
};

struct Row {
    int actual_team_count, actual_thread_count;
    int team_id, thread_id, index, call_count;
};

struct Plan {
    int effective_team_count = 0;
    int effective_thread_count = 0;
    int max_chunks = 0;
    // Inferred chunks are maximal runs of consecutive indices owned by a team.
    // Neither these boundaries nor execution order were observed by the CSV.
    bool chunk_boundaries_inferred = true;
    bool call_order_inferred = true;
    bool idle_teams_unobserved = false;
    std::vector<int> chunk_counts;
    // Slot = (team * max_chunks + chunk) * effective_thread_count + thread.
    // offsets[slot]..offsets[slot+1] indexes that slot's sorted work list.
    std::vector<int> offsets;
    std::vector<int> indices;
};

inline std::runtime_error error(const std::string& message) {
    return std::runtime_error("OpenMP profile: " + message);
}

inline std::string trim(const std::string& value) {
    const std::string whitespace = " \t\r\n";
    size_t first = value.find_first_not_of(whitespace);
    if (first == std::string::npos) return "";
    size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

inline std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result;
    size_t begin = 0;
    for (;;) {
        size_t end = line.find(',', begin);
        result.push_back(trim(line.substr(begin, end == std::string::npos ? end : end - begin)));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return result;
}

inline int integer(const std::string& value, size_t line) {
    if (value.empty()) throw error("empty numeric field at line " + std::to_string(line));
    errno = 0;
    char *end = NULL;
    long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end || parsed < INT_MIN || parsed > INT_MAX)
        throw error("invalid integer at line " + std::to_string(line) + ": " + value);
    return (int)parsed;
}

class Profile {
public:
    std::map<Key, std::vector<Row> > observations;

    Plan select(int method, int n, int iterations, int size, int requested_teams,
                int requested_threads) const {
        Key key = {method, n, iterations, size, requested_teams, requested_threads};
        std::map<Key, std::vector<Row> >::const_iterator found = observations.find(key);
        if (found == observations.end()) {
            std::ostringstream message;
            message << "no exact workload key method=" << method << " N=" << n
                    << " I=" << iterations << " M=" << size
                    << " requested B=" << requested_teams << " T=" << requested_threads;
            throw error(message.str());
        }
        const std::vector<Row>& rows = found->second;
        Plan plan;
        std::map<int, Row> owner_by_index;
        std::vector<bool> seen_teams;
        for (size_t r = 0; r < rows.size(); ++r) {
            const Row& row = rows[r];
            if (row.actual_team_count <= 0 || row.actual_thread_count <= 0)
                throw error("participant counts were not observed (zero-work profiles cannot be replayed)");
            if (!plan.effective_team_count) {
                plan.effective_team_count = row.actual_team_count;
                plan.effective_thread_count = row.actual_thread_count;
                seen_teams.resize(row.actual_team_count, false);
            } else if (plan.effective_team_count != row.actual_team_count ||
                       plan.effective_thread_count != row.actual_thread_count) {
                throw error("nonuniform team/thread counts; a rectangular HIP launch cannot replay this profile");
            }
            if (row.index < 0) continue; // Explicit metadata row, if supplied.
            if (!owner_by_index.insert(std::make_pair(row.index, row)).second)
                throw error("duplicate or conflicting owners for index " + std::to_string(row.index));
            seen_teams[row.team_id] = true;
        }
        if (!plan.effective_team_count) throw error("empty workload profile");
        const int b = plan.effective_team_count, t = plan.effective_thread_count;
        plan.idle_teams_unobserved = std::find(seen_teams.begin(), seen_teams.end(), false) != seen_teams.end();

        if (method == 6 || method == 8 || method == 9) {
            if (owner_by_index.size() != (size_t)iterations)
                throw error("worksharing profile has missing or extra iteration indices");
            for (int i = 0; i < iterations; ++i) {
                std::map<int, Row>::const_iterator owner = owner_by_index.find(i);
                if (owner == owner_by_index.end() || owner->second.call_count != 1)
                    throw error("iteration " + std::to_string(i) + " must have exactly one observed call");
            }
        } else {
            size_t expected_rows = 1;
            if (method == 5) {
                if (t != 1) throw error("scalar teams profile must have one active thread per team");
                expected_rows = (size_t)b;
            } else if (method == 10 || method == 11) {
                expected_rows = (size_t)b * (size_t)t;
            } else if (b != 1 || t != 1) {
                throw error("scalar target profile must have one observed team and thread");
            }
            if (owner_by_index.size() != expected_rows)
                throw error("profile does not cover every expected participant");
            std::vector<bool> seen_participants;
            if (expected_rows > seen_participants.max_size()) throw error("participant table size overflow");
            seen_participants.assign(expected_rows, false);
            for (std::map<int, Row>::const_iterator it = owner_by_index.begin(); it != owner_by_index.end(); ++it) {
                const Row& row = it->second;
                long long expected_index = 0;
                size_t participant = 0;
                if (method == 5) {
                    expected_index = (long long)row.team_id * size / requested_teams;
                    participant = (size_t)row.team_id;
                } else if (method == 10 || method == 11) {
                    expected_index = method == 10 ? (long long)row.team_id * (size / requested_teams) + row.thread_id :
                        ((long long)row.team_id * requested_threads + row.thread_id) % size;
                    participant = (size_t)row.team_id * t + row.thread_id;
                }
                if (row.index != expected_index || row.call_count != (method == 11 ? n : 1) ||
                    seen_participants[participant])
                    throw error("profile owner/index/call count disagrees with the original method's configured stride");
                seen_participants[participant] = true;
            }
            plan.chunk_counts.assign(b, 0);
            plan.offsets.push_back(0);
            return plan; // Only effective B/T is needed by non-worksharing methods.
        }

        // A contiguous run owned by one team forms an inferred chunk. Grouping
        // cannot recover the OpenMP implementation's actual fork/join regions.
        typedef std::vector<std::vector<int> > Chunk;
        std::vector<std::vector<Chunk> > chunks(b);
        std::vector<int> previous(b, -2);
        for (std::map<int, Row>::const_iterator it = owner_by_index.begin(); it != owner_by_index.end(); ++it) {
            const Row& row = it->second;
            if ((long long)previous[row.team_id] + 1 != row.index)
                chunks[row.team_id].push_back(Chunk(t));
            chunks[row.team_id].back()[row.thread_id].push_back(row.index);
            previous[row.team_id] = row.index;
        }
        plan.chunk_counts.resize(b);
        for (int team = 0; team < b; ++team) {
            if (chunks[team].size() > (size_t)INT_MAX) throw error("too many inferred chunks");
            plan.chunk_counts[team] = (int)chunks[team].size();
            plan.max_chunks = std::max(plan.max_chunks, plan.chunk_counts[team]);
        }
        size_t slots = (size_t)b;
        if (plan.max_chunks && slots > (size_t)(INT_MAX - 1) / (size_t)plan.max_chunks)
            throw error("work-list offset table size overflow");
        slots *= (size_t)plan.max_chunks;
        if (t && slots > (size_t)(INT_MAX - 1) / (size_t)t)
            throw error("work-list offset table size overflow");
        slots *= (size_t)t;
        plan.offsets.reserve(slots + 1);
        plan.offsets.push_back(0);
        for (int team = 0; team < b; ++team)
        for (int chunk = 0; chunk < plan.max_chunks; ++chunk)
        for (int thread = 0; thread < t; ++thread) {
            if (chunk < plan.chunk_counts[team]) {
                const std::vector<int>& work = chunks[team][chunk][thread];
                if (work.size() > (size_t)INT_MAX - plan.indices.size()) throw error("work-list index size overflow");
                plan.indices.insert(plan.indices.end(), work.begin(), work.end());
            }
            plan.offsets.push_back((int)plan.indices.size());
        }
        return plan;
    }
};

inline Profile read(const std::string& path) {
    std::ifstream input(path.c_str());
    if (!input) throw error("cannot open " + path);
    const std::string expected_header = "method_id,N,max_iter,max_array_size,requested_team_count,requested_thread_count,actual_team_count,actual_thread_count,team_id,thread_id,index,call_count";
    std::string line;
    if (!std::getline(input, line) || trim(line) != expected_header)
        throw error("unexpected CSV header in " + path);
    Profile profile;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty()) continue;
        std::vector<std::string> values = fields(line);
        if (values.size() != 12) throw error("expected 12 fields at line " + std::to_string(line_number));
        int value[12];
        for (int i = 0; i < 12; ++i) value[i] = integer(values[i], line_number);
        Key key = {value[0], value[1], value[2], value[3], value[4], value[5]};
        Row row = {value[6], value[7], value[8], value[9], value[10], value[11]};
        if (key.method_id < 0 || key.method_id > 11 || key.N <= 0 || key.max_iter < 0 ||
            key.max_array_size <= 0 || key.requested_team_count <= 0 || key.requested_thread_count <= 0 ||
            ((key.method_id == 6 || key.method_id == 8 || key.method_id == 9) && key.max_iter > key.max_array_size) ||
            (key.method_id >= 1 && key.method_id <= 4 && key.N > key.max_array_size))
            throw error("invalid workload key at line " + std::to_string(line_number));
        bool metadata = row.team_id == -1 && row.thread_id == -1 && row.index == -1 && row.call_count == 0;
        bool unknown = row.actual_team_count == -1 && row.actual_thread_count == -1;
        if (metadata) {
            if (!unknown && (row.actual_team_count <= 0 || row.actual_thread_count <= 0))
                throw error("invalid metadata participant counts at line " + std::to_string(line_number));
        } else if (row.actual_team_count <= 0 || row.actual_thread_count <= 0 ||
                   row.team_id < 0 || row.team_id >= row.actual_team_count ||
                   row.thread_id < 0 || row.thread_id >= row.actual_thread_count ||
                   row.index < 0 || row.index >= key.max_array_size || row.call_count <= 0) {
            throw error("invalid observed owner/index/count at line " + std::to_string(line_number));
        }
        profile.observations[key].push_back(row);
    }
    if (input.bad()) throw error("failed while reading " + path);
    if (profile.observations.empty()) throw error("no observations in " + path);
    return profile;
}

} // namespace microbenchmark_openmp_profile
#endif
