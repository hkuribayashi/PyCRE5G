class MOGWOSegment:
    def __init__(self, archive_max):
        self.archive_max = archive_max
        self.archive = []

    def add_solution(self, p):
        if len(self.archive) < self.archive_max:
            if len(self.archive) is 0:
                self.archive.append(p)
            else:
                list_del = []
                flag = False
                for non_dominated in self.archive:
                    if p <= non_dominated:
                        flag = True
                        list_del.append(non_dominated)
                    elif p.evaluation_f1 <= non_dominated.evaluation_f1 or p.evaluation_f2 <= non_dominated.evaluation_f2:
                        flag = True
                    else:
                        flag = False
                for element in list_del:
                    self.archive.remove(element)
                if flag:
                    self.archive.append(p)
            self.archive = list(set(self.archive))
            return False
        else:
            return True

    def is_full(self):
        if len(self.archive ) == self.archive_max:
            return True
        else:
            return False

    def __str__(self):
        str_return = ""
        for p in self.archive:
            str_return = str_return + p
        return "[" + str_return + "]"
