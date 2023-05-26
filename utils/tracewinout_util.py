import numpy as np
from ess import TraceWin

class TraceWinOutUtil():

    @classmethod
    def read_bpm_from_tracewinout(cls,tracewinout,bpm_position):
        data = TraceWin.partran(tracewinout)
        z_indices = [np.where(np.isclose(data['z(m)'], z))[0][0] for z in bpm_position]
        x_list = data['x0'][z_indices]
        y_list = data['y0'][z_indices]
        bpm_list = []
        # genrate a bpm list with[x1,y1,x2,y2....]
        [bpm_list.extend([x,y]) for x, y in zip(x_list,y_list)]
        return bpm_list

    @classmethod
    def read_current_from_tracewinout(cls,tracewinout,bpm_position):
        pass

