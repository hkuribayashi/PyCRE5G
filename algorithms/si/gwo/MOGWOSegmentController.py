import copy
import random

from algorithms.si import MOGWOSegment


class MOGWOSegmentController:

    def __init__(self, n_segments, non_dominated_solutions):
        self.segments = {}
        self.probability = {}
        self.n_segments = n_segments
        for idx in range(n_segments):
            self.segments[idx] = MOGWOSegment(15)
            self.probability[idx] = 0.0
            self.segments[idx].archive = non_dominated_solutions[idx]

    def select_leader(self):
        for key in self.segments:
            if len(self.segments[key].archive) > 0:
                self.probability[key] = 100 - len(self.segments[key].archive)
            else:
                self.probability[key] = 0
        selected_key_segment = random.choices(list(self.segments.keys()), weights=list(self.probability.values()), k=1)
        key = selected_key_segment[-1]
        p = copy.deepcopy(self.segments[key].archive[0])
        self.segments[key].archive.pop(0)
        return p, key

    def add_leader(self, segment_id, leader):
        self.segments[segment_id].archive.append(copy.deepcopy(leader))

    def update(self, non_dominated_list):
        for p in non_dominated_list:
            flag = True
            if not self.is_full():
                while flag:
                    key = random.randint(0, self.n_segments - 1)
                    flag = self.segments[key].add_solution(p)
                return True
            return False

    def get_archive_size(self):
        archive_size = 0
        for key in self.segments:
            archive_size += len(self.segments[key].archive)
        return archive_size

    def get_archived_solutions(self):
        archived_solutions = []
        for key in self.segments:
            archived_solutions.append(self.segments[key].archive)
        return archived_solutions

    def is_full(self):
        flag = True
        for key in self.segments:
            if not self.segments[key].is_full():
                flag = False
        return flag
