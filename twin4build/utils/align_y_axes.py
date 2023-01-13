import numpy as np

def calculate_ticks(ax, nticks, round_to=0.1):


    # min_max_array = np.array([[min(line.get_ydata()),max(line.get_ydata())] for line in ax.lines])
    # print(min_max_array)
    # min_ = np.min(min_max_array)
    # max_ = np.max(min_max_array)

    # ax.set_ylim([min_,max_])

    lower = np.floor(ax.get_ylim()[0]/round_to)*round_to
    upper = np.ceil(ax.get_ylim()[1]/round_to)*round_to
    total_ticks = (upper-lower)/round_to
    rem = np.remainder(total_ticks,(nticks - 1))
    new_upper = upper - rem*round_to# + round_to*(nticks-1)
    total_ticks = (new_upper-lower)/round_to
    rem = np.remainder(total_ticks,(nticks - 1))
    ticks = np.linspace(lower, new_upper, nticks)
    return ticks

def alignYaxes(axes_list, nticks_list, round_to_list, yoffset_list):
    for ax,nticks,round_to in zip(axes_list,nticks_list,round_to_list):
        ticks = calculate_ticks(ax, nticks=nticks, round_to=round_to)
        ax.set_yticks(ticks)
        ax.set_ylim([ticks[0],ticks[-1]])

    ybound_list = [ax.get_ylim() for ax in axes_list]
    yoffset_ybound_master_list = [[yoffset,ybound] for yoffset,ybound in zip(yoffset_list,ybound_list) if yoffset is not None][0]
    yoffset_master = yoffset_ybound_master_list[0]
    ydiff_master = yoffset_ybound_master_list[1][1]-yoffset_ybound_master_list[1][0]
    ydiff_list = [ax.get_ylim()[1]-ax.get_ylim()[0] for ax in axes_list]
    yoffset_new_list = [((ydiff*yoffset_master)/(ydiff_master+2*yoffset_master))/(1-(2*yoffset_master)/(ydiff_master+2*yoffset_master)) for ydiff in ydiff_list]

    for ax,yoffset_new in zip(axes_list,yoffset_new_list):
        ax.set_ylim([ax.get_ylim()[0]-yoffset_new,ax.get_ylim()[1]+yoffset_new])

