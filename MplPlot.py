from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from sci_calculation import *
import numpy as np

class GBPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=18, height=12, dpi=100, data=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        #self.fig.set_tight_layout(True)
        self.gs=self.fig.add_gridspec(2,2,left=0.1, wspace=0.1, width_ratios=[0.5,0.5])
        super(GBPlot, self).__init__(self.fig)
        self.lineay = []
        self.lineaz = []
        self.lineacomb = []
        self.initplotxyz()
        self.annots=[]
        self.annot_mode = 'Single'


    def initplotxyz(self):
        try:
            for ax in self.get_all_axe_list():
                ax.remove()
            for line in self.get_all_line_list():
                line.remove()


        except:
            pass
        self.lineay = []
        self.lineaz = []
        self.lineacomb = []
        ax1 = self.fig.add_subplot(self.gs[0,0])
        ax2 = self.fig.add_subplot(self.gs[0,1])
        ax3 = self.fig.add_subplot(self.gs[1,:])

        self.axe_ay=ax1
        self.axe_az=ax2
        self.axe_comb=ax3
        self.axe_ay.plot([0.05,0.2,1,4],[3,2,2,2],'r',linewidth=3,label='Allowable ay')
        self.axe_ay.plot([0.05, 0.2, 1, 4], [-3, -2, -2, -2], 'r', linewidth=3)
        self.axe_az.plot([0,1,2,3,4],[6,6,4,4,4],'r',linewidth=3,label='Allowable az')
        self.axe_az.plot([0,0.5,2,3,4],[-2,-1.5,-1.5,-1.5,-1.5],'r',linewidth=3)
        self.axe_comb.plot([-1.8,-1.62,-0.54,0,1.8,5.4,6],[0,0.6,1.8,2,1.8,0.6,0],'yellow',linewidth=3,label='dt=0.2s')
        self.axe_comb.plot([-1.8, -1.62, -0.54, 0, 1.8, 5.4, 6], [0, -0.6, -1.8, -2, -1.8, -0.6, 0], 'yellow', linewidth=3,
                           )
        self.axe_comb.plot([-1.9,-1.71,-0.57,0,1.8,5.4,6],[0,0.741,2.22,2.47,2.22,0.741,0],'orange',linewidth=3,label='dt=0.1s')
        self.axe_comb.plot([-1.9, -1.71, -0.57, 0, 1.8, 5.4, 6], [0, -0.741, -2.22, -2.47, -2.22, -0.741, 0], 'orange',
                           linewidth=3)
        self.axe_comb.plot([-1.95,-1.755,-0.585,0,1.8,5.4,6],[0,0.9,2.7,3,2.7,0.9,0],'r',linewidth=3,label='dt=0.05s')
        self.axe_comb.plot([-1.95, -1.755, -0.585, 0, 1.8, 5.4, 6], [0, -0.9, -2.7, -3, -2.7, -0.9, 0], 'r', linewidth=3,
                           )

        self.axe_ay.set_xlim(0,4)
        self.axe_ay.set_xlabel('dt (s)')
        self.axe_ay.set_ylim(-5,5)
        self.axe_ay.set_ylabel('ay')
        self.axe_az.set_xlim(0,4)
        self.axe_az.set_xlabel('dt (s)')
        self.axe_az.set_ylim(-2,6)
        self.axe_az.set_ylabel('az')
        self.axe_comb.set_xlim(-2,6)
        self.axe_comb.set_ylim(-3.2,3.2)
        self.axe_comb.set_xlabel('az')
        self.axe_comb.set_ylabel('ay')




        for ax in self.get_all_axe_list():
            ax.legend()
        self.fig.canvas.draw_idle()
    def removelines(self):

        try:
            self.lineay.pop().remove()
            self.lineay.pop().remove()
            self.lineaz.pop().remove()
            self.lineaz.pop().remove()
            self.lineacomb.pop().remove()

        except Exception as e:
            #print (self.lineay)
            pass
    def addplotxyz(self, data):
        try:
            for annot in self.annots:
                annot.remove()
        except:
            pass


        self.removelines()
        d0py=max(0,np.max(data.iloc[:,2]))
        d0ny=min(0,np.min(data.iloc[:,2]))
        d0pz=max(1,np.max(data.iloc[:,3]))
        d0nz=min(1,np.min(data.iloc[:,3]))
        d1p,d1n=ProcessGB(data,5)
        d2p,d2n=ProcessGB(data,25)
        d3p,d3n=ProcessGB(data,100)
        d4p,d4n=ProcessGB(data,250)
        d5p,d5n=ProcessGB(data,500)
        d6p,d6n=ProcessGB(data,1000)
        d7p,d7n=ProcessGB(data,2000)
        d08p,d08n = ProcessGB(data,40)
        d10p, d10n = ProcessGB(data, 50)
        d30p, d30n = ProcessGB(data, 150)
        d40p, d40n = ProcessGB(data, 200)
        z_array=data.iloc[:,3].values
        y_array=data.iloc[:,2].values
        p1=[d0py,d1p[0],d2p[0],d08p[0],d10p[0],d3p[0],d30p[0],d40p[0],d4p[0],d5p[0],d6p[0],d7p[0]]
        p2=[d0ny,d1n[0],d2n[0],d08n[0],d10n[0],d3n[0],d30n[0],d40n[0],d4n[0],d5n[0],d6n[0],d7n[0]]
        p3=[d0pz,d1p[1],d2p[1],d08p[1],d10p[1],d3p[1],d30p[1],d40p[1],d4p[1],d5p[1],d6p[1],d7p[1]]
        p4=[d0nz,d1n[1],d2n[1],d08n[1],d10n[1],d3n[1],d30n[1],d40n[1],d4n[1],d5n[1],d6n[1],d7n[1]]

        x_axis_gb = [0.002,0.01,0.05,0.08,0.1,0.2,0.3,0.4,0.5,1,2,4]
        self.lineay.append(self.axe_ay.plot(x_axis_gb,p1,'-',color='b',label='Measured ay(+)')[0])
        self.lineay.append(self.axe_ay.plot(x_axis_gb,p2,'-',color='k',label='Measured ay( - )')[0])
        self.lineaz.append(self.axe_az.plot(x_axis_gb,p3,'-',color='b',label='Measured az(+)')[0])
        self.lineaz.append(self.axe_az.plot(x_axis_gb,p4,'-',color='k',label='Measured az( - )')[0])
        self.lineacomb.append(self.axe_comb.plot(z_array,y_array,'.-',color='c',label='Measured Data')[0])
        try:
            for ax in self.get_all_axe_list():
                ax.legend()
        except:
            pass


        self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
    def setlinewidth(self, linewidth=0.6):
        for line in self.get_all_line_list():
            line.set_linewidth(linewidth)
    def get_all_axe_list(self):
        axes_list=[self.axe_ay, self.axe_az, self.axe_comb]
        return axes_list
    def get_all_line_list(self):
        line_list = []
        for axs in [self.lineay, self.lineaz, self.lineacomb]:
            for ax in axs:
                line_list.append(ax)

        #print('line_list_num = ' + str(len(line_list)))
        return line_list
    def update_annot(self, ax, line, ind):
        try:

            x, y = line.get_data()
            annot = ax.annotate("", xy=(x[ind["ind"][0]], y[ind["ind"][0]]), xytext=(10+x[ind["ind"][0]], 10+y[ind["ind"][0]]), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="b", alpha=0.1),
                                    arrowprops=dict(arrowstyle="->"))

            if ax == self.axe_comb:
                text = "t = {}\naz = {}\nay = {}".format(round(ind["ind"][0]*0.002,3), x[ind["ind"][0]],y[ind["ind"][0]])
            else:
                text = "t = {}\namp = {}".format(x[ind["ind"][0]], y[ind["ind"][0]])
            annot.set_text(text)
            annot.set_visible(True)
            annot.set_fontsize(8)
            self.annots.append(annot)
        except Exception as e:
            print(e)

    def onclick(self, event):
        if self.annot_mode == "Single":
            try:
                for annot in self.annots:
                    annot.set_visible(False)
                self.annots=[]
            except:
                pass
        if event.inaxes in self.get_all_axe_list():
            for line in self.get_all_line_list():
                try:
                    cont, ind = line.contains(event)
                    if cont:
                        self.update_annot(event.inaxes, line, ind)
                        self.fig.canvas.draw_idle()
                        break

                except Exception as e:
                    pass

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=18, height=12, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        #self.fig.set_tight_layout(True)
        super(MplCanvas, self).__init__(self.fig)
        # self.initplotxyz()

        self.linex = []
        self.liney = []
        self.linez = []
        self.annots=[]
        self.annot_mode = 'Single'
        self.initplotxyz()

    def initplotxyz(self, list_data=None):
        try:
            for ax in self.get_all_axe_list():
                ax.remove()
            for line in self.get_all_line_list():
                line.remove()
        except:
            pass

        self.linex = []
        self.liney = []
        self.linez = []
        self.axes_x = self.fig.add_subplot(311)
        self.axes_y = self.fig.add_subplot(312, sharex= self.axes_x)
        self.axes_z = self.fig.add_subplot(313, sharex = self.axes_x)

        i=0
        ylabels=['Fore/Aft', 'Lateral', 'Vertical']
        for ax in self.get_all_axe_list():
            ax.set_xlabel('time (s)')
            ax.set_ylabel(ylabels[i])
            i+=1

        self.fig.canvas.draw_idle()

    def removelines(self, plot_order):
        plot_order=plot_order
        try:
            self.linex.pop(plot_order).remove()
            self.liney.pop(plot_order).remove()
            self.linez.pop(plot_order).remove()
            self.axes_x.pop(plot_order).remove()
            self.axes_y.pop(plot_order).remove()
            self.axes_z.pop(plot_order).remove()
            self.fig.canvas.draw_idle()
            #print('line {}'.format(len(self.get_all_line_list())))
        except Exception as e:
            pass

    def addplotxyz(self, list_data, plot_order, color='b'):
        try:
            for annot in self.annots:
                annot.remove()
        except:
            pass



        line1, = self.axes_x.plot(list_data.iloc[:, 0], list_data.iloc[:, 1], color=color)
        self.linex.append(line1)

        line2, = self.axes_y.plot(list_data.iloc[:, 0], list_data.iloc[:, 2], color=color)
        self.liney.append(line2)

        line3, = self.axes_z.plot(list_data.iloc[:, 0], list_data.iloc[:, 3], color=color)
        self.linez.append(line3)
        self.setlinewidth()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)



    def addplotxyz_fitmax(self, list_data, plot_order, offset, color='b'):
        try:
            for annot in self.annots:
                annot.remove()
        except:
            pass


        t=(list_data.iloc[:, 0].values - offset*0.002).round(decimals=4)

        line1, = self.axes_x.plot(t, list_data.iloc[:, 1], color=color)
        self.linex.append(line1)

        line2, = self.axes_y.plot(t, list_data.iloc[:, 2], color=color)
        self.liney.append(line2)

        line3, = self.axes_z.plot(t, list_data.iloc[:, 3], color=color)
        self.linez.append(line3)
        self.setlinewidth()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)




    def setlinewidth(self, linewidth=0.6):
        for line in self.get_all_line_list():
            line.set_linewidth(linewidth)

    def get_all_axe_list(self):
        return [self.axes_x, self.axes_y, self.axes_z]

    def get_all_line_list(self):
        line_list = []
        for axs in [self.linex, self.liney, self.linez]:
            for ax in axs:
                line_list.append(ax)
        return line_list



    def update_annot(self, ax, line, ind):
        x, y = line.get_data()
        annot = ax.annotate("", xy=(x[ind["ind"][0]], y[ind["ind"][0]]), xytext=(10, 10), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="b", alpha=0.1),
                                arrowprops=dict(arrowstyle="->"))

        text = "t = {}\namp = {}".format(x[ind["ind"][0]], y[ind["ind"][0]])
        annot.set_text(text)
        annot.set_fontsize(8)
        annot.set_visible(True)
        self.annots.append(annot)

    def onclick(self, event):
        if self.annot_mode == "Single":
            try:
                for annot in self.annots:
                    annot.set_visible(False)
                self.annots=[]
            except:
                pass
        if event.inaxes in self.get_all_axe_list():
            for line in self.get_all_line_list():
                try:
                    cont, ind = line.contains(event)
                    if cont:
                        self.update_annot(event.inaxes, line, ind)
                        self.fig.canvas.draw_idle()
                        break
                except Exception as e:
                    pass


class AccZonePlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=18, height=12, dpi=100, data=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        #self.fig.set_tight_layout(True)
        super(AccZonePlot, self).__init__(self.fig)

        self.initplotxyz()
        self.annots=[]
        self.annot_mode = 'Single'


    def initplotxyz(self):
        self.removelines()
        self.axe = self.fig.add_subplot(111)
        self.axe.fill([-0.2, -0.2, 0.2, 0.2, 1.7, 1.7, -0.2], [1.2, 0.7, 0.7, 0.2, 0.2, 1.2, 1.2], color='greenyellow')
        self.axe.fill([-0.7, -0.7, 0.2, 0.2, -0.2, -0.2, -0.7], [1.2, 0.2, 0.2, 0.7, 0.7, 1.2, 1.2], color='yellow')
        self.axe.fill([-1.2, -1.2, -0.7, -0.7, -1.2], [1.2, 0.2, 0.2, 1.2, 1.2], color='orange')
        self.axe.fill([-1.7, -1.7, 1.7, 1.7, 0.7, -0.7, -1.2, -1.2, -1.7], [1.2, 0, 0, -0.2, -0.2, 0.2, 0.2, 1.2, 1.2],
                 color='salmon')
        self.axe.fill([-0.7, 1.7, 1.7, 0, -0.7], [0.2, 0.2, 0, 0, 0.2], color='orange')
        self.axe.fill([-1.7, 0, 0.7, 1.7, 1.7, -1.7, -1.7], [0, 0, -0.2, -0.2, -0.3, -0.3, 0], 'red')
        self.axe.fill([-0.2, 0, -0.2], [0, 0, 0.4 / 7], 'red')
        self.axe.fill([0, 0.2, 0.2], [0, 0, -0.4 / 7], 'red')
        self.axe.set_xlim(-1.7, 1.7)
        self.axe.set_ylim(-0.3, 1.2)
        self.axe.set_xlabel('Front <=> Back')
        self.axe.set_ylabel('Up <=> Down')
        self.axe.set_title('Acceleration Zones')
        self.axe.text(0.7, 0.7, 'Zone 1')
        self.axe.text(-0.4, 0.5, 'Zone 2')
        self.axe.text(-1.1, 0.5, 'Zone 3')
        self.axe.text(0.7, 0.05, 'Zone 3')
        self.axe.text(1, -0.12, 'Zone 4')
        self.axe.text(-0.3, -0.2, 'Zone 5')
        self.axe.text(-1.6, 0.5, 'Zone 4')

        self.fig.canvas.draw_idle()

    def removelines(self):
        try:
            self.line.remove()
        except:
            pass


    def addplotxyz(self, data):
        self.removelines()
        try:
            for annot in self.annots:
                annot.remove()
        except Exception as e:
            pass

        self.line, = self.axe.plot(data.iloc[:, 1], data.iloc[:, 3], '.-', color='blue')
        self.fig.canvas.draw_idle()



    def update_annot(self, ax, ind):
        x, y = self.line.get_data()
        annot = ax.annotate("", xy=(x[ind["ind"][0]], y[ind["ind"][0]]), xytext=(10+x[ind["ind"][0]], 10+y[ind["ind"][0]]), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="b", alpha=0.1),
                                arrowprops=dict(arrowstyle="->"))

        text = "t = {}s\nx = {}\nz = {}".format(round(ind["ind"][0]*0.002,3), x[ind["ind"][0]],y[ind["ind"][0]])
        annot.set_text(text)
        annot.set_fontsize(8)
        annot.set_visible(True)
        self.annots.append(annot)

    def onclick(self, event):
        if self.annot_mode == "Single":
            try:
                for annot in self.annots:
                    annot.set_visible(False)
                self.annots=[]
            except:
                pass
        if event.inaxes == self.axe:
            try:
                cont, ind = self.line.contains(event)
                if cont:
                    self.update_annot(event.inaxes, ind)
                    self.fig.canvas.draw_idle()

            except Exception as e:
                pass



class ASTMPlot(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=18, height=12, dpi=100, data=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        #self.fig.set_tight_layout(True)

        super(ASTMPlot, self).__init__(self.fig)
        self.lineax = []
        self.lineay = []
        self.lineaz = []
        self.lineeggxy=[]
        self.lineeggxz=[]
        self.lineeggyz=[]
        self.annot_mode = 'Single'
        self.initplotxyz()
        self.annots=[]


    def initplotxyz(self):
        try:
            for ax in self.get_all_axe_list():
                ax.remove()
            for line in self.get_all_line_list():
                line.remove()
        except:
            pass
        self.lineax = []
        self.lineay = []
        self.lineaz = []
        self.lineeggxy=[]
        self.lineeggxz=[]
        self.lineeggyz=[]
        self.axe_ax = self.fig.add_subplot(231)
        self.axe_ay = self.fig.add_subplot(232)
        self.axe_az = self.fig.add_subplot(233)
        self.axe_eggxy = self.fig.add_subplot(234)
        self.axe_eggxz = self.fig.add_subplot(235)
        self.axe_eggyz = self.fig.add_subplot(236)

        self.axe_eggxy.set_xlabel('Front <=> Back')
        self.axe_eggxy.set_ylabel('Left <=> Right')
        self.axe_eggxz.set_xlabel('Front <=> Back')
        self.axe_eggxz.set_ylabel('Up <=> Down')
        self.axe_eggyz.set_xlabel('Left <=> Right')
        self.axe_eggyz.set_ylabel('Up <=> Down')
        self.axe_ax.set_xlabel('dt (s)')
        self.axe_ax.set_ylabel('ax')
        self.axe_ay.set_xlabel('dt (s)')
        self.axe_ay.set_ylabel('ay')
        self.axe_az.set_xlabel('dt (s)')
        self.axe_az.set_ylabel('az')

        self.axe_ax.set_xlim(0, 14)
        self.axe_ay.set_xlim(0, 14)
        self.axe_ay.set_ylim(0, 3.2)
        self.axe_az.set_xlim(0, 14)
        self.axe_eggxy.set_xlim(-2.1, 6.1)
        self.axe_eggxy.set_ylim(-3.2, 3.2)
        self.axe_eggxz.set_xlim(-2.1, 6.1)
        self.axe_eggxz.set_ylim(-2.2, 6.2)
        self.axe_eggyz.set_xlim(-3.1, 3.1)
        self.axe_eggyz.set_ylim(-2.2, 6.2)

        #for ax in self.get_all_axe_list():
            #ax.legend()

        self.fig.canvas.draw_idle()

    def removelines(self):

        try:
            self.lineax.pop().remove()
            self.lineax.pop().remove()
            self.lineay.pop().remove()
            self.lineay.pop().remove()
            self.lineaz.pop().remove()
            self.lineaz.pop().remove()
            self.lineeggxy.pop().remove()
            self.lineeggxz.pop().remove()
            self.lineeggyz.pop().remove()

        except Exception as e:
            pass

    def addplotxyz(self, data, restraint, cond, input_height):
        try:
            for annot in self.annots:
                annot.remove()
        except:
            pass

        self.removelines()













        d0p, d0n = ProcessASTM(data, 5)

        d1p, d1n = ProcessASTM(data, 10)
        d2p, d2n = ProcessASTM(data, 25)
        d3p, d3n = ProcessASTM(data, 50)
        d4p, d4n = ProcessASTM(data, 250)
        d5p, d5n = ProcessASTM(data, 500)
        d6p, d6n = ProcessASTM(data, 2000)
        d7p, d7n = ProcessASTM(data, 100)

        d8p, d8n = ProcessASTM(data, 7000)
        d9p, d9n = ProcessASTM(data, 1000)
        d10p, d10n = ProcessASTM(data, 2500)
        d77p, d77n = ProcessASTM(data, 3000)
        d99p, d99n = ProcessASTM(data, 3500)
        d88p, d88n = ProcessASTM(data, 4000)
        d111p, d111n = ProcessASTM(data, 4500)
        d03p, d03n = ProcessASTM(data, 150)
        d04p, d04n = ProcessASTM(data, 200)


        x_axis = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4, 5, 6, 7,8, 9, 14]
        self.lineax.append(self.axe_ax.plot(x_axis,
                [d0p[0], d1p[0], d2p[0], d3p[0], d7p[0], d03p[0],d04p[0],d4p[0], d5p[0], d9p[0], d6p[0], d10p[0], d77p[0], d99p[0],d88p[0],
                 d111p[0], d8p[0]], 'b-', label='Measured ax (+)')[0])
        self.lineax.append(self.axe_ax.plot(x_axis,
                [d0n[0], d1n[0], d2n[0], d3n[0], d7n[0], d03n[0],d04n[0],d4n[0], d5n[0], d9n[0], d6n[0], d10n[0], d77n[0], d99n[0],d88n[0],
                 d111n[0], d8n[0]], 'g-', label='Measured ax (-)')[0])
        self.lineay.append(self.axe_ay.plot(x_axis,
                [d0p[1], d1p[1], d2p[1], d3p[1], d7p[1],d03p[1],d04p[1], d4p[1], d5p[1], d9p[1], d6p[1], d10p[1], d77p[1], d99p[1],d88p[1],
                 d111p[1], d8p[1]], 'b-', label='Measured ay (Right)')[0])
        self.lineay.append(self.axe_ay.plot(x_axis,
                [-d0n[1], -d1n[1], -d2n[1], -d3n[1], -d7n[1], -d03n[1],-d04n[1],-d4n[1], -d5n[1], -d9n[1], -d6n[1], -d10n[1], -d77n[1],
                 -d99n[1],-d88n[1], -d111n[1], -d8n[1]], 'g-', label='Measured ay (Left)')[0])
        self.lineaz.append(self.axe_az.plot(x_axis,
                [d0p[2], d1p[2], d2p[2], d3p[2], d7p[2],d03p[2],d04p[2], d4p[2], d5p[2], d9p[2], d6p[2], d10p[2], d77p[2],  d99p[2],d88p[2],
                 d111p[2], d8p[2]], 'b-', label='Measured az (Down)')[0])
        self.lineaz.append(self.axe_az.plot(x_axis,
                [d0n[2], d1n[2], d2n[2], d3n[2], d7n[2], d03n[2],d04n[2],d4n[2], d5n[2], d9n[2], d6n[2], d10n[2], d77n[2], d99n[2],d88n[2],
                 d111n[2], d8n[2]], 'g-', label='Measured az ( Up )')[0])

        self.lineeggxy.append(self.axe_eggxy.plot(data.iloc[:, 1], data.iloc[:, 2], 'c.-', label='Measured')[0])
        self.lineeggxz.append(self.axe_eggxz.plot(data.iloc[:, 1], data.iloc[:, 3], 'c.-', label='Measured')[0])
        self.lineeggyz.append(self.axe_eggyz.plot(data.iloc[:, 2], data.iloc[:, 3], 'c.-', label='Measured')[0])

        try:
            for ax in self.get_all_axe_list():
                ax.legend()
        except:
            pass



        try:
            self.addDisneyStd(restype=restraint, cond=cond, height=input_height)
        except Exception as e:
            pass

        self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    def setlinewidth(self, linewidth = 0.6):
        for line in self.get_all_line_list():
            line.set_linewidth(linewidth)

    def get_all_axe_list(self):
        axes_list=[self.axe_ax, self.axe_ay, self.axe_az, self.axe_eggxy,self.axe_eggxz, self.axe_eggyz]
        return axes_list

    def get_all_line_list(self):
        line_list = []
        for axs in [self.lineax,self.lineay, self.lineaz, self.lineeggxy, self.lineeggxz, self.lineeggyz]:
            for ax in axs:
                line_list.append(ax)

        #print('line_list_num = ' + str(len(line_list)))
        return line_list


    def update_annot(self, ax, line, ind):
        x, y = line.get_data()
        annot = ax.annotate("", xy=(x[ind["ind"][0]], y[ind["ind"][0]]), xytext=(10+x[ind["ind"][0]], 10+y[ind["ind"][0]]), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="b", alpha=0.1),
                                arrowprops=dict(arrowstyle="->"))

        if ax == self.axe_eggxy:
            text = "t = {}\nx = {}\ny = {}".format(round(ind["ind"][0] * 0.002, 3), x[ind["ind"][0]], y[ind["ind"][0]])
        elif ax == self.axe_eggxz:
            text = "t = {}\nx = {}\nz = {}".format(round(ind["ind"][0] * 0.002, 3), x[ind["ind"][0]], y[ind["ind"][0]])
        elif ax == self.axe_eggyz:
            text = "t = {}\ny = {}\nz = {}".format(round(ind["ind"][0] * 0.002, 3), x[ind["ind"][0]], y[ind["ind"][0]])
        else:
            text = "t = {}\namp = {}".format(x[ind["ind"][0]], y[ind["ind"][0]])
        annot.set_text(text)
        annot.set_fontsize(8)
        annot.set_visible(True)
        self.annots.append(annot)

    def onclick(self, event):
        if self.annot_mode == "Single":
            try:
                for annot in self.annots:
                    annot.set_visible(False)
                self.annots=[]
            except:
                pass
        if event.inaxes in self.get_all_axe_list():
            for line in self.get_all_line_list():
                try:
                    cont, ind = line.contains(event)
                    if cont:
                        self.update_annot(event.inaxes, line, ind)
                        self.fig.canvas.draw_idle()
                        break

                except Exception as e:
                    pass

    def addDisneyStd(self,cond,restype, height):
        try:
            for lines in [self.lined1, self.lined2, self.lined3, self.lined4, self.lined5, self.lined6]:
                for line in lines:
                    line.remove()
        except:
            pass

        if cond == 'E-Stop':
            cond = 1.25
        elif cond == "Normal":
            cond = 1
        else:
            cond = 1.25


        self.lined1=[]
        self.lined2=[]
        self.lined3=[]
        self.lined4=[]
        self.lined5=[]
        self.lined6=[]
        frontASTM = [-2, -2, -1.5, -1.5]
        backASTM = [6, 6, 6, 4, 4, 3, 3, 2.5, 2.5]
        lrASTM = [3, 3, 3, 2, 2]
        upASTM = [-2, -2, -1.5, -1.5, -1.2, -1.2]
        downASTM = [6, 6, 6, 4, 4, 3, 3, 2, 2]
        ht = float(height)
        coefX, coefZ= cond * coef(ht, 'x'), cond * coef(ht, 'z')

        x3, y3 = eggXY(2 * coef(ht, 'x'), 6 * coef(ht, 'x'), 3 * coef(ht, 'x'))
        x4, y4 = eggXZ(2 * coef(ht, 'z'), 6 * coef(ht, 'z'), 2 * coef(ht, 'z'), 6 * coef(ht, 'z'))
        x5, y5 = eggYZ(3 * coef(ht, 'z'), 2 * coef(ht, 'z'), 6 * coef(ht, 'z'))
        self.lined1.append(self.axe_ax.plot([0.2, 0.5, 14], [-2,-1.5, -1.5], 'r', linewidth=3, label='Allowable ax')[0])
        self.lined1.append(self.axe_ax.plot(np.array([0.2, 1, 2, 4, 5, 11.8, 12, 14]), np.array([6, 6, 4, 4, 3, 3, 2.5, 2.5]) * coef(ht, 'x'), 'r', linewidth=3)[0])
        self.lined2.append(self.axe_ay.plot(np.array([0.2, 1, 2, 14]), np.array([3, 3, 2, 2])* coef(ht, 'x'), 'r', linewidth=3, label='Allowable ay')[0])
        self.lined3.append(self.axe_az.plot(np.array([0.2, 0.5, 4, 7, 14]), np.array([-2, -1.5, -1.5, -1.2, -1.2])* coef(ht, 'x'), 'r', linewidth=3, label='Allowable az')[0])
        self.lined3.append(self.axe_az.plot(np.array([0.2, 1, 2, 4, 5, 11.8, 12, 14]), np.array([6, 6, 4, 4, 3, 3, 2, 2]) * coef(ht, 'x'), 'r', linewidth=3)[0])
        self.lined4.append( self.axe_eggxy.plot(x3, y3, 'r--', label='ASTM (0.2s)')[0])
        self.lined4.append(self.axe_eggxy.plot([-2, 6], [0, 0], [0, 0], [3, -3], color='k', linewidth=3)[0])
        self.lined5.append(self.axe_eggxz.plot(x4, y4, 'r--', label='ASTM (0.2s)')[0])
        self.lined5.append(self.axe_eggxz.plot([-2, 6], [0, 0], [0, 0], [6, -2], color='k', linewidth=3)[0])
        self.lined6.append(self.axe_eggyz.plot(x5, y5, 'r--', label='ASTM (0.2s)')[0])
        self.lined6.append(self.axe_eggyz.plot([-3, 3], [0, 0], [0, 0], [-2, 6], color='k', linewidth=3)[0])

        try:
            for ax in self.get_all_axe_list():
                ax.legend()

        except:
            pass

        if restype == 'Upper Body':

            xa, ya = eggXY(2 * cond * coef(ht, 'x'), 3.6 * cond * coef(ht, 'x'), 3 * cond * coef(ht, 'x'))
            xaa, yaa = eggXY(1.6 * cond * coef(ht, 'x'), 3.6 * cond * coef(ht, 'x'),
                             2.4 * cond * coef(ht, 'x'))
            xb, yb = eggXZ(2 * cond * coef(ht, 'z'), 3.6 * cond * coef(ht, 'z'), 2 * cond * coef(ht, 'z'),
                           5 * cond * coef(ht, 'z'))
            xbb, ybb = eggXZ(1.6 * cond * coef(ht, 'z'), 3.6 * cond * coef(ht, 'z'),
                             1.4 * cond * coef(ht, 'z'), 4.8 * cond * coef(ht, 'z'))
            xc, yc = eggYZ(3 * cond * coef(ht, 'z'), 2 * cond * coef(ht, 'z'), 5 * cond * coef(ht, 'z'))
            xcc, ycc = eggYZ(2.4 * cond * coef(ht, 'z'), 1.4 * cond * coef(ht, 'z'),
                             4.8 * cond * coef(ht, 'z'))
            a31 = np.maximum(frontASTM, np.array([-2, -1.6, -1.2, -1.2]) * coefX)
            a32 = np.minimum(backASTM, np.array([3.6, 3.6, 3.6, 2.5, 2.5, 2, 2, 2, 2]) * coefX)
            a33 = np.minimum(lrASTM, np.array([3, 2.4, 2.4, 1.6, 1.6]) * coefX)
            a34 = np.maximum(upASTM, np.array([-2, -1.4, -1, -1, -0.7, -0.7]) * coefZ)
            a35 = np.minimum(downASTM, np.array([5, 4.8, 4.8, 3.4, 3.4, 2.6, 2.6, 1.8, 1.8]) * coefZ)
            
            self.lined4.append(self.axe_eggxy.plot(xa, ya, 'k', label='Upper Body (0s)')[0])
            self.lined4.append(self.axe_eggxy.plot(xaa, yaa, 'k--', label='Upper Body (0.2s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xb, yb, 'k', label='Upper Body (0s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xbb, ybb, 'k--', label='Upper Body (0.2s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xc, yc, 'k', label='Upper Body (0s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xcc, ycc, 'k--', label='Upper Body (0.2s)')[0])

        elif restype == 'Group Lower Body':

            xa, ya = eggXY(1.7 * cond * coef(ht, 'x'), 2.5 * cond * coef(ht, 'x'), 2.4 * cond * coef(ht, 'x'))
            xaa, yaa = eggXY(1.4 * cond * coef(ht, 'x'), 2.5 * cond * coef(ht, 'x'),
                             2.1 * cond * coef(ht, 'x'))
            xb, yb = eggXZ(1.7 * cond * coef(ht, 'z'), 2.5 * cond * coef(ht, 'z'), 2 * cond * coef(ht, 'z'),
                           3.5 * cond * coef(ht, 'z'))
            xbb, ybb = eggXZ(1.4 * cond * coef(ht, 'z'), 2.5 * cond * coef(ht, 'z'), 1 * cond * coef(ht, 'z'),
                             3 * cond * coef(ht, 'z'))
            xc, yc = eggYZ(2.4 * cond * coef(ht, 'z'), 2 * cond * coef(ht, 'z'), 3.5 * cond * coef(ht, 'z'))
            xcc, ycc = eggYZ(2.1 * cond * coef(ht, 'z'), 1 * cond * coef(ht, 'z'), 3 * cond * coef(ht, 'z'))
            a31 = np.maximum(frontASTM, np.array([-2, -1.6, -1.2, -1.2]) * coefX)
            a32 = np.minimum(backASTM, np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2, 2, 2, 2]) * coefX)
            a33 = np.minimum(lrASTM, np.array([2.4, 2.1, 2.1, 1.4, 1.4]) * coefX)
            a34 = np.maximum(upASTM, np.array([-1, 0, 0.2, 0.2, 0.2, 0.2]) * coefZ)
            a35 = np.minimum(downASTM, np.array([4.5, 4, 4, 3.1, 3.1, 2.4, 2.4, 1.7, 1.7]) * coefZ)

            self.lined4.append(self.axe_eggxy.plot(xa, ya, 'k', label='Group Lower Body (0s)')[0])
            self.lined4.append(self.axe_eggxy.plot(xaa, yaa, 'k--', label='Group Lower Body (0.2s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xb, 1 + yb, 'k', label='Group Lower Body (0s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xbb, 1 + ybb, 'k--', label='Group Lower Body (0.2s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xc, 1 + yc, 'k', label='Group Lower Body (0s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xcc, 1 + ycc, 'k--', label='Group Lower Body (0.2s)')[0])

        elif restype == 'Individual Lower Body':

            xa, ya = eggXY(1.8 * cond * coef(ht, 'x'), 2.5 * cond * coef(ht, 'x'), 2.6 * cond * coef(ht, 'x'))
            xaa, yaa = eggXY(1.5 * cond * coef(ht, 'x'), 2.5 * cond * coef(ht, 'x'),
                             2.2 * cond * coef(ht, 'x'))
            xb, yb = eggXZ(1.8 * cond * coef(ht, 'z'), 2.5 * cond * coef(ht, 'z'), 1.8 * cond * coef(ht, 'z'),
                           4.8 * cond * coef(ht, 'z'))
            xbb, ybb = eggXZ(1.5 * cond * coef(ht, 'z'), 2.5 * cond * coef(ht, 'z'),
                             1.2 * cond * coef(ht, 'z'), 4.5 * cond * coef(ht, 'z'))
            xc, yc = eggYZ(2.6 * cond * coef(ht, 'z'), 1.8 * cond * coef(ht, 'z'), 4.8 * cond * coef(ht, 'z'))
            xcc, ycc = eggYZ(2.2 * cond * coef(ht, 'z'), 1.2 * cond * coef(ht, 'z'),
                             4.5 * cond * coef(ht, 'z'))

            a31 = np.maximum(frontASTM, np.array([-1.8, -1.5, -1.1, -1.1]) * coefX)
            a32 = np.minimum(backASTM, np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2, 2, 2, 2]) * coefX)
            a33 = np.minimum(lrASTM, np.array([2.6, 2.2, 2.2, 1.5, 1.5]) * coefX)
            a34 = np.maximum(upASTM, np.array([-1.8, -1.2, -0.9, -0.9, -0.6, -0.6]) * coefZ)
            a35 = np.minimum(downASTM, np.array([4.8, 4.5, 4.5, 3.2, 3.2, 2.5, 2.5, 1.8, 1.8]) * coefZ)

            self.lined4.append(self.axe_eggxy.plot(xa, ya, 'k', label='Individual Lower Body (0s)')[0])
            self.lined4.append(self.axe_eggxy.plot(xaa, yaa, 'k--', label='Individual Lower Body (0.2s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xb, yb, 'k', label='Individual Lower Body (0s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xbb, ybb, 'k--', label='Individual Lower Body (0.2s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xc, yc, 'k', label='Individual Lower Body (0s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xcc, ycc, 'k--', label='Individual Lower Body (0.2s)')[0])



        elif restype == 'No Restraint' or restype == 'Convenience Restraint':

            xa, ya = eggXY(1.5 * cond * coef(ht, 'x'), 2.5 * cond * coef(ht, 'x'), 1.8 * cond * coef(ht, 'x'))
            xaa, yaa = eggXY(1.2 * cond * coef(ht, 'x'), 2.5 * cond * coef(ht, 'x'),
                             1.2 * cond * coef(ht, 'x'))
            xb, yb = eggXZ(1.5 * cond * coef(ht, 'z'), 2.5 * cond * coef(ht, 'z'), 1.2 * cond * coef(ht, 'z'),
                           3 * cond * coef(ht, 'z'))
            xbb, ybb = eggXZ(1.2 * cond * coef(ht, 'z'), 2.5 * cond * coef(ht, 'z'),
                             0.8 * cond * coef(ht, 'z'), 2.8 * cond * coef(ht, 'z'))
            xc, yc = eggYZ(1.8 * cond * coef(ht, 'z'), 1.2 * cond * coef(ht, 'z'), 3 * cond * coef(ht, 'z'))
            xcc, ycc = eggYZ(1.2 * cond * coef(ht, 'z'), 0.8 * cond * coef(ht, 'z'),
                             2.8 * cond * coef(ht, 'z'))

            a31 = np.maximum(frontASTM, np.array([-1.5, -1.2, -0.7, -0.7]) * coefX)
            a32 = np.minimum(backASTM, np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2, 2, 2, 2]) * coefX)
            a33 = np.minimum(lrASTM, np.array([1.8, 1.2, 1.2, 0.7, 0.7]) * coefX)
            a34 = np.maximum(upASTM, np.array([-0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * coefZ)
            a35 = np.minimum(downASTM, np.array([4, 3.8, 3.8, 2.8, 2.8, 2.2, 2.2, 1.6, 1.6]) * coefZ)

            self.lined4.append(self.axe_eggxy.plot(xa, ya, 'k', label='No/Conv Restraint (0s)')[0])
            self.lined4.append(self.axe_eggxy.plot(xaa, yaa, 'k--', label='No/Conv Restraint (0.2s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xb, 1 + yb, 'k', label='No/Conv Restraint (0s)')[0])
            self.lined5.append(self.axe_eggxz.plot(xbb, 1 + ybb, 'k--', label='No/Conv Restraint (0.2s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xc, 1 + yc, 'k', label='No/Conv Restraint (0s)')[0])
            self.lined6.append(self.axe_eggyz.plot(xcc, 1 + ycc, 'k--', label='No/Conv Restraint (0.2s)')[0])

        self.lined1.append(self.axe_ax.plot([0, 0.2, 0.5, 14], a31, 'r', linewidth=2, label=restype)[0])
        self.lined1.append(self.axe_ax.plot([0, 0.2, 1, 2, 4, 5, 11.8, 12, 14], a32, 'r',linewidth=2,)[0])
        self.lined2.append(self.axe_ay.plot([0, 0.2, 1, 2, 14], a33, 'r',linewidth=2, label=restype)[0])
        self.lined3.append(self.axe_az.plot([0, 0.2, 0.5, 4, 7, 14], a34, 'r',linewidth=2,)[0])
        self.lined3.append(self.axe_az.plot([0, 0.2, 1, 2, 4, 5, 11.8, 12, 14], a35, 'r', linewidth=2,label=restype)[0])


        for ax in self.get_all_axe_list():
            ax.legend()