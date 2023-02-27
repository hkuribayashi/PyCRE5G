class Slice:
    def __init__(self, hetnet):
        self.selected_bs = hetnet.list_bs
        self.priority_ues_weight = hetnet.list_bs[-1].hetnet.env.priority_ues_weight
        self.ordinary_ues_weight = hetnet.list_bs[-1].hetnet.env.priority_ues_weight
        self.evaluation = hetnet.evaluation['satisfaction']
        self.ue_list = hetnet.ue_list
