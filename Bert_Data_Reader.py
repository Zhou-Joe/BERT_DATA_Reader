
# coding: utf-8

# In[1]:


import math
import matplotlib
import time
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.figure import Figure
import tkinter.filedialog as filedialog
from pathlib import Path
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.image as image


import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter
from scipy.signal import freqs

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    if cutoff==500:
        return data
    else:
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
    return y


def ProcessASTM(data,dt):
    
    if dt>999:
        x=int(dt/50)
    else:
        x=5
    data=data.loc[::x,:]
    dt=int(max(dt/x,1))
    x_array=list(data.iloc[:,1].values)
    y_array=list(data.iloc[:,2].values)                                      
    z_array=list(data.iloc[:,3].values)
    m=len(y_array)

    maxp=np.max(list(map(lambda i:[max(min(x_array[i:i+dt]),0),max(min(y_array[i:i+dt]),0),max(min(z_array[i:i+dt]),1)],range(m-dt))),axis=0)
    maxn=np.min(list(map(lambda i:[min(max(x_array[i:i+dt]),0),min(max(y_array[i:i+dt]),0),min(max(z_array[i:i+dt]),1)],range(m-dt))),axis=0)
    return maxp,maxn

def ProcessGB(data,dt):
    if dt>999:
        x=int(dt/50)
    else:
        x=5
    data=data.loc[::x,:]
    dt=int(max(dt/x,1))
    y_array=list(data.iloc[:,2].values)                                      
    z_array=list(data.iloc[:,3].values)
    m=len(y_array)

    maxp=np.max(list(map(lambda i:[max(min(y_array[i:i+dt]),0),max(min(z_array[i:i+dt]),1)],range(m-dt))),axis=0)
    maxn=np.min(list(map(lambda i:[min(max(y_array[i:i+dt]),0),min(max(z_array[i:i+dt]),1)],range(m-dt))),axis=0)
    return maxp,maxn

  
class OverlayData:
    def __init__(self):
        self.rawdata=pd.DataFrame()
        self.offset=None
        
        
    def from_file(self,path):
        self.rawdata=pd.read_table(path,header=None,skiprows=[0,1,2,3,4,5])
        self.rawdata.columns=['time','x','y','z']
        self.get_offset()
        self.filename = path
        
    def from_lap(self,data):
        self.rawdata=pd.DataFrame(data.iloc[5:,:].values)
        self.rawdata.columns=['time','x','y','z']
        self.get_offset()
        
    def get_offset(self):
        if self.rawdata.shape[0] > 500:
            temp=self.rawdata.iloc[0:499,:].values
            self.offset=np.average(temp,axis=0)
            self.offset[3]-=1
        

class AccData:
    def __init__(self,path, filefolder=None):
        try:
            self.path=path
            self.pathfolder=Path(path).resolve().parent
            self.rawdata=pd.read_table(path,header=None,skiprows=[0]).iloc[:,0:6]
        except:
            self.rawdata=path
            self.path=Path(filefolder).resolve().parent
            self.pathfolder=self.path
        if  self.rawdata.shape[1]<=5:
            self.rawdata.drop([0,1,2,3,4],inplace=True)
            self.rawdata.drop(0,axis=1,inplace=True)
            self.rawdata.reset_index(drop=True,inplace=True)
            self.rawdata['x']=self.rawdata.iloc[:,0]
            self.rawdata['y']=self.rawdata.iloc[:,1]
            self.rawdata['z']=self.rawdata.iloc[:,2]
            
        self.laps=1  #####by default
        self.datasize=self.rawdata.shape[0]
        self.hastitle=False
        self.issep=False
        self.GBdata=[pd.DataFrame for i in range (self.laps)]
        self.ASTMdata=[pd.DataFrame for i in range (self.laps)]
        self.rawdata.columns=['x_GB','y_GB','z_GB','x_ASTM','y_ASTM','z_ASTM']
        time=pd.DataFrame(np.arange(0,self.datasize*0.002,0.002),columns=['time'])
        self.rawdata['time']=time
        #self.rawdata['x_GB']=self.rawdata['x_GB'].values*(-1)
        self.get_offset()
        
    def changelaps(self,n):
        self.laps=n
        self.GBdata=[pd.DataFrame for i in range (self.laps)]
        self.ASTMdata=[pd.DataFrame for i in range (self.laps)]
        
    def get_offset(self):
        if self.rawdata.shape[0] > 500:
            temp=self.rawdata.iloc[0:499,:].values
            self.offset=np.average(temp,axis=0)
            self.offset[2]-=1
            self.offset[5]-=1
        
        
    def datasep(self,laps_start_time):
        laps_start_time=np.append(laps_start_time,self.datasize*0.002)
        self.hastitle=False
            
            
        for i in range(self.laps):
            self.GBdata[i]=self.rawdata.loc[laps_start_time[i]/0.002:(laps_start_time[i+1]-0.002)/0.002,['time','x_GB','y_GB','z_GB']]
                
            self.GBdata[i]['time']=self.GBdata[i]['time'].values-laps_start_time[i]
            self.ASTMdata[i]=self.rawdata.loc[laps_start_time[i]/0.002:(laps_start_time[i+1]-0.002)/0.002,['time','x_ASTM','y_ASTM','z_ASTM']]
            self.ASTMdata[i]['time']=self.ASTMdata[i]['time'].values-laps_start_time[i]

        
        
    def addtitle(self):
        if self.hastitle is False:
            self.GBdata_t=self.GBdata
            self.ASTMdata_t=self.ASTMdata
            for i in range(self.laps):
                self.ASTMdata_t[i].loc[-5,'time']='Initial Pitch Angle: 0'
                self.ASTMdata_t[i].loc[-4,'time']='Initial Roll Angle: 0'
                self.ASTMdata_t[i].loc[-3,'time']='Initial Yaw Angle: 0'
                self.ASTMdata_t[i].loc[-2,'time']='SARCcolumnIDs: Time ASTM_Accel-X ASTM_Accel-Y ASTM_Accel-Z'
                self.ASTMdata_t[i].loc[-1,'time']=np.nan
                self.GBdata_t[i].loc[-5,'time']='Initial Pitch Angle: 0'
                self.GBdata_t[i].loc[-4,'time']='Initial Roll Angle: 0'
                self.GBdata_t[i].loc[-3,'time']='Initial Yaw Angle: 0'
                self.GBdata_t[i].loc[-2,'time']='SARCcolumnIDs: Time GB_Accel-X GB_Accel-Y GB_Accel-Z'
                self.GBdata_t[i].loc[-1,'time']=np.nan
                self.GBdata_t[i].index+=5
                self.GBdata_t[i]=self.GBdata[i].sort_index()
                self.ASTMdata_t[i].index+=5
                self.ASTMdata_t[i]=self.ASTMdata[i].sort_index()
            self.hastitle=True
        else:
            print ('Title already added')
            
            
    def writefile(self,fname):
        for i in range (self.laps):
            pathstrA=str(self.pathfolder)+'/'+fname+'_ASTM_Lap'+str(i+1)+'.sup'
            pathstrG=str(self.pathfolder)+'/'+fname+'_GB_Lap'+str(i+1)+'.sup'
            self.ASTMdata_t[i].to_csv(pathstrA,sep='\t',header=False,index=False)
            self.GBdata_t[i].to_csv(pathstrG,sep='\t',header=False,index=False)
   
        

    
    

    
    
    


# In[2]:


global t,x,y,z
t=[]
x=[]
y=[]
z=[]
f=Figure(figsize=(10,8),dpi=100)
a=f.add_subplot(311)
b=f.add_subplot(312)
c=f.add_subplot(313)
ao=f.add_subplot(311)
bo=f.add_subplot(312)
co=f.add_subplot(313)


a.set_ylabel('Fore/Aft')
b.set_ylabel('Lateral')
c.set_ylabel('Vertical')
c.set_xlabel('time (s)')
f.tight_layout()
a.set_xlim(0)

f2=Figure(figsize=(10,8),dpi=100)
aa=f2.add_subplot(222)
bb=f2.add_subplot(223)
cc=f2.add_subplot(221)
dd=f2.add_subplot(224)

def append4(x1,x2,x3,x4):
    x=np.append(x1,x2)
    x=np.append(x,x3)
    x=np.append(x,x4)
    return x



def eggXY(x1,x2,y_max):
    x1=min(x1,2)
    y_max=min(y_max,3)
    
    x11=np.cos(np.arange(0,math.pi*0.5,0.01))*x2
    x22=np.cos(np.arange(math.pi*0.5,math.pi,0.01))*x1
    x33=np.cos(np.arange(math.pi,math.pi*1.5,0.01))*x1
    x44=np.cos(np.arange(math.pi*1.5,math.pi*2+0.01,0.01))*x2
    y11=np.sin(np.arange(0,math.pi*0.5,0.01))*y_max
    y22=np.sin(np.arange(math.pi*0.5,math.pi,0.01))*y_max
    y33=np.sin(np.arange(math.pi,math.pi*1.5,0.01))*y_max
    y44=np.sin(np.arange(math.pi*1.5,math.pi*2+0.01,0.01))*y_max
    
    return append4(x11,x22,x33,x44),append4(y11,y22,y33,y44)


def eggXZ(x1,x2,y1,y2):
    x1=min(x1,2)
    y1=min(y1,2)
    y2=min(y2,6)
    x11=np.cos(np.arange(0,math.pi*0.5,0.01))*x2
    x22=np.cos(np.arange(math.pi*0.5,math.pi,0.01))*x1
    x33=np.cos(np.arange(math.pi,math.pi*1.5,0.01))*x1
    x44=np.cos(np.arange(math.pi*1.5,math.pi*2+0.01,0.01))*x2
    y11=np.sin(np.arange(0,math.pi*0.5,0.01))*y2
    y22=np.sin(np.arange(math.pi*0.5,math.pi,0.01))*y2
    y33=np.sin(np.arange(math.pi,math.pi*1.5,0.01))*y1
    y44=np.sin(np.arange(math.pi*1.5,math.pi*2+0.01,0.01))*y1
    
    return append4(x11,x22,x33,x44),append4(y11,y22,y33,y44)

def eggYZ(x,y1,y2):
    y1=min(2,y1)
    y2=min(6,y2)
    x=min(3,x)
    x11=np.cos(np.arange(0,math.pi*0.5,0.01))*x
    x22=np.cos(np.arange(math.pi*0.5,math.pi,0.01))*x
    x33=np.cos(np.arange(math.pi,math.pi*1.5,0.01))*x
    x44=np.cos(np.arange(math.pi*1.5,math.pi*2+0.01,0.01))*x
    y11=np.sin(np.arange(0,math.pi*0.5,0.01))*y2
    y22=np.sin(np.arange(math.pi*0.5,math.pi,0.01))*y2
    y33=np.sin(np.arange(math.pi,math.pi*1.5,0.01))*y1
    y44=np.sin(np.arange(math.pi*1.5,math.pi*2+0.01,0.01))*y1
    
    return append4(x11,x22,x33,x44),append4(y11,y22,y33,y44)

def PlotDuration(a3,b3,c3,coefX,coefZ,restype):
        frontASTM=[-2,-2,-1.5,-1.5]
        backASTM=[6,6,6,4,4,3,3,2.5,2.5]
        lrASTM=[3,3,3,2,2]
        upASTM=[-2,-2,-1.5,-1.5,-1.2,-1.2]
        downASTM=[6,6,6,4,4,3,3,2,2]
        if restype=='Individual Lower Body':
            a31=np.maximum(frontASTM,np.array([-1.8,-1.5,-1.1,-1.1])*coefX)
            a32=np.minimum(backASTM,np.array([2.5,2.5,2.5,2.5,2.5,2,2,2,2])*coefX)
            a33=np.minimum(lrASTM,np.array([2.6,2.2,2.2,1.5,1.5])*coefX)
            a34=np.maximum(upASTM,np.array([-1.8,-1.2,-0.9,-0.9,-0.6,-0.6])*coefZ)
            a35=np.minimum(downASTM,np.array([4.8,4.5,4.5,3.2,3.2,2.5,2.5,1.8,1.8])*coefZ)
            
            
            
        elif restype=='Upper Body':
            a31=np.maximum(frontASTM,np.array([-2,-1.6,-1.2,-1.2])*coefX)
            a32=np.minimum(backASTM,np.array([3.6,3.6,3.6,2.5,2.5,2,2,2,2])*coefX)
            a33=np.minimum(lrASTM,np.array([3,2.4,2.4,1.6,1.6])*coefX)
            a34=np.maximum(upASTM,np.array([-2,-1.4,-1,-1,-0.7,-0.7])*coefZ)
            a35=np.minimum(downASTM,np.array([5,4.8,4.8,3.4,3.4,2.6,2.6,1.8,1.8])*coefZ)
            
            
        elif restype=='Group Lower Body':
            a31=np.maximum(frontASTM,np.array([-2,-1.6,-1.2,-1.2])*coefX)
            a32=np.minimum(backASTM,np.array([2.5,2.5,2.5,2.5,2.5,2,2,2,2])*coefX)
            a33=np.minimum(lrASTM,np.array([2.4,2.1,2.1,1.4,1.4])*coefX)
            a34=np.maximum(upASTM,np.array([-1,0,0.2,0.2,0.2,0.2])*coefZ)
            a35=np.minimum(downASTM,np.array([4.5,4,4,3.1,3.1,2.4,2.4,1.7,1.7])*coefZ)
            
        elif restype=='Convenience Restraint' or restype=='No Restraint':
            a31=np.maximum(frontASTM,np.array([-1.5,-1.2,-0.7,-0.7])*coefX)
            a32=np.minimum(backASTM,np.array([2.5,2.5,2.5,2.5,2.5,2,2,2,2])*coefX)
            a33=np.minimum(lrASTM,np.array([1.8,1.2,1.2,0.7,0.7])*coefX)
            a34=np.maximum(upASTM,np.array([-0.2,0.2,0.2,0.2,0.2,0.2])*coefZ)
            a35=np.minimum(downASTM,np.array([4,3.8,3.8,2.8,2.8,2.2,2.2,1.6,1.6])*coefZ)
                 
    
        a3.plot([0,0.2,0.5,14],a31,'k',label=restype)
        a3.plot([0,0.2,1,2,4,5,11.8,12,14],a32,'k')
        b3.plot([0,0.2,1,2,14],a33,'k',label=restype)
        c3.plot([0,0.2,0.5,4,7,14],a34,'k')
        c3.plot([0,0.2,1,2,4,5,11.8,12,14],a35,'k',label=restype)


def coef(height,condition):
    if condition=='x'or condition=='y':
        if height<=32:
            return 0.52
        elif height<=48:
            return height*0.03-0.44
        else:
            return 1


        
    if condition=='z':
        if height<=32:
            return 0.6
        elif height<=48:
            return height*0.025-0.2
        else:
            return 1        
        
        

f3=Figure(figsize=(8,7),dpi=100)
a3=f3.add_subplot(231)
b3=f3.add_subplot(232)
c3=f3.add_subplot(233)
d3=f3.add_subplot(234)
e3=f3.add_subplot(235)
ff3=f3.add_subplot(236)


def initializeF3():
    a3.clear()
    b3.clear()
    c3.clear()
    d3.clear()
    e3.clear()
    ff3.clear()
    a3.plot([0.2,0.5,14],[-2,-1.5,-1.5],'r',linewidth=3,label='Allowable ax')
    a3.plot([0.2,1,2,4,5,11.8,12,14],[6,6,4,4,3,3,2.5,2.5],'r',linewidth=3)
    b3.plot([0.2,1,2,14],[3,3,2,2],'r',linewidth=3,label='Allowable ay')
    c3.plot([0.2,0.5,4,7,14],[-2,-1.5,-1.5,-1.2,-1.2],'r',linewidth=3,label='Allowable az')
    c3.plot([0.2,1,2,4,5,11.8,12,14],[6,6,4,4,3,3,2,2],'r',linewidth=3)
    aa.set_title('ax Duration Limit')
    bb.set_title('ay Duration Limit')
    cc.set_title('az Duration Limit')

    a3.set_xlabel('dt (s)')
    a3.set_ylabel('ax')
    b3.set_xlabel('dt (s)')
    b3.set_ylabel('ay')
    c3.set_xlabel('dt (s)')
    c3.set_ylabel('az')
        
    a3.set_xlim(0,14)
    b3.set_xlim(0,14)        
    d3.set_xlim(-2.1,6.1)
    d3.set_ylim(-3.2,3.2)
    e3.set_xlim(-2.1,6.1)
    e3.set_ylim(-2.2,6.2)
    ff3.set_xlim(-3.1,3.1)
    ff3.set_ylim(-2.2,6.2)
    c3.set_xlim(0,14)
    
    
    x3,y3=eggXY(2,6,3)
    x4,y4=eggXZ(2,6,2,6)
    x5,y5=eggYZ(3,2,6)
    
    
    
    d3.plot(x3,y3,'r--',label='ASTM (0.2s)')
    d3.plot([-2,6],[0,0],'k',linewidth=3)
    d3.plot([0,0],[3,-3],'k',linewidth=3) 
    e3.plot(x4,y4,'r--',label='ASTM (0.2s)')
    e3.plot([-2,6],[0,0],'k',linewidth=3)
    e3.plot([0,0],[6,-2],'k',linewidth=3)
    d3.legend()
    ff3.plot(x5,y5,'r--',label='ASTM (0.2s)')
    ff3.plot([-3,3],[0,0],'k',linewidth=3)
    ff3.plot([0,0],[-2,6],'k',linewidth=3)
    d3.set_xlabel('Front <=> Back')
    d3.set_ylabel('Left <=> Right') 
    e3.set_xlabel('Front <=> Back')
    e3.set_ylabel('Up <=> Down') 
    ff3.set_xlabel('Left <=> Right')
    ff3.set_ylabel('Up <=> Down')    
    e3.legend()
    d3.legend()
    ff3.legend()


f4=Figure(figsize=(8,8),dpi=100)
f4a=f4.add_subplot(111)

def initializeF4():
    f4a.clear()
    f4a.fill([-0.2,-0.2,0.2,0.2,1.7,1.7,-0.2],[1.2,0.7,0.7,0.2,0.2,1.2,1.2],color='greenyellow')
    f4a.fill([-0.7,-0.7,0.2,0.2,-0.2,-0.2,-0.7],[1.2,0.2,0.2,0.7,0.7,1.2,1.2],color='yellow')
    f4a.fill([-1.2,-1.2,-0.7,-0.7,-1.2],[1.2,0.2,0.2,1.2,1.2],color='orange')
    f4a.fill([-1.7,-1.7,1.7,1.7,0.7,-0.7,-1.2,-1.2,-1.7],[1.2,0,0,-0.2,-0.2,0.2,0.2,1.2,1.2],color='salmon')
    f4a.fill([-0.7,1.7,1.7,0,-0.7],[0.2,0.2,0,0,0.2],color='orange')
    f4a.fill([-1.7,0,0.7,1.7,1.7,-1.7,-1.7],[0,0,-0.2,-0.2,-0.3,-0.3,0],'red')
    f4a.fill([-0.2,0,-0.2],[0,0,0.4/7],'red')
    f4a.fill([0,0.2,0.2],[0,0,-0.4/7],'red')
    f4a.set_xlim(-1.7,1.7)
    f4a.set_ylim(-0.3,1.2)
    f4a.set_xlabel('Front <=> Back')
    f4a.set_ylabel('Up <=> Down')
    f4a.set_title('Acceleration Zones')
    f4a.text(0.7,0.7,'Zone 1')
    f4a.text(-0.4,0.5,'Zone 2')
    f4a.text(-1.1,0.5,'Zone 3')
    f4a.text(0.7,0.05,'Zone 3')
    f4a.text(1,-0.12,'Zone 4')
    f4a.text(-0.3,-0.2,'Zone 5')
    f4a.text(-1.6,0.5,'Zone 4')


def initializeF2():
    aa.clear()
    bb.clear()
    cc.clear()
    dd.clear()
    aa.loglog([0.01,0.2,1,4],[5,2,2,2],'r',linewidth=3,label='Allowable ay')
    bb.plot([0,1,2,3,4],[6,6,4,4,4],'r',linewidth=3,label='Allowable az')
    bb.plot([0,0.5,2,3,4],[-2,-1.5,-1.5,-1.5,-1.5],'r',linewidth=3)
    cc.imshow(image.imread('CoordGB8408.jpg'))
    dd.plot([-1.8,-1.62,-0.54,0,1.8,5.4,6],[0,0.6,1.8,2,1.8,0.6,0],'yellow',linewidth=3,label='dt=0.05s')
    dd.plot([-1.9,-1.71,-0.57,0,1.8,5.4,6],[0,0.741,2.22,2.47,2.22,0.741,0],'orange',linewidth=3,label='dt=0.1s')
    dd.plot([-1.95,-1.755,-0.585,0,1.8,5.4,6],[0,0.9,2.7,3,2.7,0.9,0],'r',linewidth=3,label='dt=0.2s')
    aa.set_title('ay Duration Limit')
    bb.set_title('az Duration Limit')
    cc.set_title('Patron Coordinate System per GB 8408')
    dd.set_title('Typical Allowable az-|ay| Combination')

    aa.set_xlabel('dt (s)')
    aa.set_ylabel('|ay|')
    bb.set_xlabel('dt (s)')
    bb.set_ylabel('az')
    dd.set_xlabel('az')
    dd.set_ylabel('|ay|')
        
    cc.axis('off')
    aa.set_xlim(0.01,4)
    aa.set_ylim(0.01,10)        
    
    bb.set_xlim(0.001,4)
    bb.set_ylim(-2,7)
    dd.set_xlim(-2,6)
    dd.set_ylim(0,3.5)

    
    
    
    
    f2.tight_layout()









# In[ ]:


def popupmsg(msg,title):
    popup = tk.Toplevel(bg='white')
    popup.wm_title(title)
    pframe=tk.Frame(popup,bg='white')
    pframe.pack(anchor=tk.CENTER)
    label = tk.Label(pframe,bg='white',text=msg)
    label.pack(anchor=tk.CENTER, fill=tk.BOTH, pady=10)
    B1 = ttk.Button(popup, text="OK", command = popup.destroy)
    B1.pack()
    popup.mainloop()
        

    
readme='\n1. Default plot is per ASTM without filter\n2. Input Cutoff Frequency before view ASTM/GB data\n3. Data Calibration: Subtracted mean of first 500 rows\n4. Filter Type: 4th Order Butterworth Filter\n5. GB Standard per GB8408-2018\n6. ASTM Standard per F2291-13\n'


class gui(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('1280x800')
        self.config(bg='pink')
        tk.Tk.iconbitmap(self, bitmap="Icon.ico")
        tk.Tk.wm_title(self, "Bert Data Reader")
        bottom=tk.Frame(self,bg='pink1')
        bottom.pack(side=tk.BOTTOM,fill=tk.BOTH)
        #container.grid_columnconfigure(4, weight=4)
        
        self.status=tk.Label(bottom,text='Import any raw data for preview...',bg='pink1',anchor='w')
        self.status.pack(side=tk.LEFT,anchor='w')
        self.path=tk.Label(bottom,text='',bg='pink1',anchor='e')
        self.path.pack(side=tk.RIGHT,anchor='e')
        self.about='\nVersion 2.1\n1. Export file to sup format\n2. Added NewtonViewer Transformer Function\n\nDeveloped by Python 3\nRide and Show Engineering\nShanghai Disney Resort'
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open File", command = lambda: self.OpenandPreview())
        filemenu.add_separator()
        filemenu.add_command(label="Cut Data by Laps", command=lambda: self.DataCut())
        filemenu.add_command(label="Export Cut Data", command=lambda: self.SaveData())
        filemenu.add_separator()
        filemenu.add_command(label="NewtonViewer Transformer", command=lambda: self.newtonviewer_helper())
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=lambda: self.destroy())



        menubar.add_cascade(label="File", menu=filemenu)
        
        overlaymenu = tk.Menu(menubar,tearoff=0)
        overlaymenu.add_command(label="Load Overlay from File", command = lambda: self.Loadoverlay('file'))     
        overlaymenu.add_command(label="Load Overlay by Lap", command = lambda: self.Loadoverlay('lap'))
        overlaymenu.add_separator()
        overlaymenu.add_command(label="Remove Overlay", command = lambda: self.RemoveOverlay())
        menubar.add_cascade(label="Edit", menu=overlaymenu)
        self.hasoverlay=False
        self.overlaynormstatus=0
        self.overlayformat='ASTM'
        stdlaymenu = tk.Menu(menubar,tearoff=0)
        stdlaymenu.add_command(label="View GB Compliance",command = lambda: self.GenerateGB())
        stdlaymenu.add_command(label="Plot Acceleration Zones",command = lambda: self.AccZone())
        stdlaymenu.add_separator()
        stdlaymenu.add_command(label="View ASTM Compliance", command = lambda: self.GenerateASTM())
        menubar.add_cascade(label="Standards", menu=stdlaymenu)



        aboutmenu = tk.Menu(menubar,tearoff=0)
        aboutmenu.add_command(label="Readme", command = lambda: popupmsg(readme,'Readme'))
        aboutmenu.add_separator()
        aboutmenu.add_command(label="Info", command = lambda: popupmsg(self.about,'About'))
        menubar.add_cascade(label="About", menu=aboutmenu)
        
        tk.Tk.config(self, menu=menubar)
        #*****************************************frame1 on container for filter 
        frame=tk.Frame(self,bg='white')
        frame.pack(side=tk.TOP,fill=tk.X,expand=False)
        
        frame1=tk.Frame(frame,bg='white')
        frame1.pack(side=tk.LEFT,expand=False)
        lbfilter=tk.Label(frame1,text='LPF Cutoff Frequency (Hz): ',bg='white')
        lbfilter.grid(row=0,column=0)
        self.etfilter=tk.Entry(frame1,width=4)
        self.etfilter.grid(row=0,column=1)

        btGBfilter=ttk.Button(frame1,text='  Filter/\n Update',command=lambda: self.FilterShow(self.dataformat))
        btGBfilter.grid(row=0,column=2,rowspan=3,sticky='ns')
        btRemovefilter=ttk.Button(frame1,text='Reset',command=lambda: self.Preview('reset'))
        btRemovefilter.grid(row=2,column=3,columnspan=1,sticky='we')
        lblaps=tk.Label(frame1,text='Lap #: (Default: 0)',bg='white')
        self.etlaps=tk.Entry(frame1,width=4)
        lblaps.grid(row=1,column=0)
        self.etlaps.grid(row=1,column=1)
        btNorm=ttk.Button(frame1,text='Calibrate Data',command=lambda: self.ShowNorm())
        btNorm.grid(row=0,column=3,sticky='nwse')
        btUnNorm=ttk.Button(frame1,text='Undo Calibrate',command=lambda: self.ShowUnNorm())
        btUnNorm.grid(row=1,column=3)
        option=tk.StringVar()
        radio1=tk.Radiobutton(frame1,text='GB',var=option,value='GB',command=lambda:self.RadioFunc('GB','dataformat'),bg='white')
        radio1.grid(row=2,column=0)
        radio1.deselect()
        radio2=tk.Radiobutton(frame1,text='ASTM',var=option,value='ASTM',command=lambda:self.RadioFunc('ASTM','dataformat'),bg='white')
        radio2.grid(row=2,column=1)
        radio2.deselect()
        
        
        
        
        frame3=tk.Frame(frame,bg='white')
        frame3.pack(side=tk.RIGHT,expand=False)
        
        im=tk.PhotoImage(file='SHDRLogo.png',master=self)
        lb=tk.Label(frame1,bg='white')
        lb.grid(row=0,column=4,rowspan=1,padx=40)
        imcanvas=tk.Label(frame3,image=im,bg='white')
        imcanvas.image = im
        imcanvas.grid(row=0,column=5,rowspan=3,padx=30)
  
        


        
        frame3=tk.Frame(frame,bg='white')
        frame3.pack(side=tk.LEFT,expand=False)
        
        lbfit=tk.Label(frame3,text='Shift by :',bg='white')
        lbfit.grid(row=0,column=0,sticky='w')
        self.etfit=tk.Entry(frame3,width=8)
        self.etfit.grid(row=0,column=1,sticky='e')
        lbfit2=tk.Label(frame3,text=' (s)',bg='white')
        lbfit2.grid(row=0,column=2,sticky='w')        
        
        btfit=ttk.Button(frame3,text='OK',width=3,command=lambda:self.ShowShift(1))
        btfit.grid(row=0,column=2,sticky='e')
        btmaxfitx=ttk.Button(frame3,text='Fit Max X',width=8,command=lambda:self.ShowShift(2))
        btmaxfitx.grid(row=1,column=0)
        btmaxfity=ttk.Button(frame3,text='Fit Max Y',width=8,command=lambda:self.ShowShift(3))
        btmaxfity.grid(row=1,column=1)
        btmaxfitz=ttk.Button(frame3,text='Fit Max Z',width=8,command=lambda:self.ShowShift(4))
        btmaxfitz.grid(row=1,column=2) 
        vtlb=tk.Label(frame3,bg='white')
        vtlb.grid(row=0,column=3,columnspan=1,sticky='we')
                
        
        
        
        
        
        
        
        
        #*********************************************frame2 on container for max/min value        
        
        
        
        
        
        
        
        frame2=tk.Frame(frame,bd=1,relief=tk.GROOVE,bg='white')
        frame2.pack(side=tk.LEFT,expand=False)
        lbmax=tk.Label(frame2,text='Max',anchor='e',bg='white')
        lbmax.grid(row=0,column=1)
        lbmin=tk.Label(frame2,text='Min',anchor='e',bg='white')
        lbmin.grid(row=0,column=2)
        lbx=tk.Label(frame2,text='X-Axis',anchor='e',bg='white')
        lbx.grid(row=1,column=0)
        lby=tk.Label(frame2,text='Y-Axis',anchor='e',bg='white')
        lby.grid(row=2,column=0)
        lbz=tk.Label(frame2,text='Z-Axis',anchor='e',bg='white')
        lbz.grid(row=3,column=0)
        self.lbxmax=tk.Label(frame2,text='',anchor='e',bg='white')
        self.lbxmax.grid(row=1,column=1,padx=20)
        self.lbxmin=tk.Label(frame2,text='',anchor='e',bg='white')
        self.lbxmin.grid(row=1,column=2)
        self.lbymax=tk.Label(frame2,text='',anchor='e',bg='white')
        self.lbymax.grid(row=2,column=1,padx=20)
        self.lbymin=tk.Label(frame2,text='',anchor='e',bg='white')
        self.lbymin.grid(row=2,column=2)
        self.lbzmax=tk.Label(frame2,text='',anchor='e',bg='white')
        self.lbzmax.grid(row=3,column=1)
        self.lbzmin=tk.Label(frame2,text='',anchor='e',bg='white')
        self.lbzmin.grid(row=3,column=2,padx=20)
        
        olbmax=tk.Label(frame2,text='Overlay Max',anchor='e',bg='white')
        olbmax.grid(row=0,column=3)
        olbmin=tk.Label(frame2,text='Overlay Min',anchor='e',bg='white')
        olbmin.grid(row=0,column=4,padx=10)
        self.olbxmax=tk.Label(frame2,text='',anchor='e',bg='white')
        self.olbxmax.grid(row=1,column=3,padx=10)
        self.olbxmin=tk.Label(frame2,text='',anchor='e',bg='white')
        self.olbxmin.grid(row=1,column=4)
        self.olbymax=tk.Label(frame2,text='',anchor='e',bg='white')
        self.olbymax.grid(row=2,column=3,padx=10)
        self.olbymin=tk.Label(frame2,text='',anchor='e',bg='white')
        self.olbymin.grid(row=2,column=4)
        self.olbzmax=tk.Label(frame2,text='',anchor='e',bg='white')
        self.olbzmax.grid(row=3,column=3)
        self.olbzmin=tk.Label(frame2,text='',anchor='e',bg='white')
        self.olbzmin.grid(row=3,column=4,padx=10)
        
        
        #*********************************************************************************************
                

        
        
        
        self.dataformat='ASTM'
        self.Normstatus=0
        #******************************************************Canvas
        container=tk.Frame(self)
        container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)
        self.canvas = FigureCanvasTkAgg(f, container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
        
        toolbar = NavigationToolbar2Tk(self.canvas, container)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
        #***************************************************

        
        
        
    def AccZone(self):
        try:
            acczone.destroy()
        except:
            pass
        initializeF4()
        data=self.filtered_data.iloc[::10,:]
        x_zone=data.iloc[:,1].values
        z_zone=data.iloc[:,3].values
        f4a.plot(x_zone,z_zone,'x-',color='blue')
        acczone=tk.Toplevel(bg='white')
        acczone.wm_title('Acceleration Zones')
        bottom=tk.Frame(acczone)
        bottom.pack(side=tk.BOTTOM,fill=tk.BOTH)
        msg=tk.Label(bottom,text='0.02s duration per point for easy identification',bg='pink1',anchor='w')
        msg.pack(fill=tk.BOTH)
        canvas = FigureCanvasTkAgg(f4, acczone)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    
        
        toolbar = NavigationToolbar2Tk(canvas, acczone)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def MoveAxis(self,x,y,z,xx,yy,zz,method):
        
        
        if method==1:
            dt=float(self.etfit.get())
        
            
        elif method==2:
            t1=np.argmax(np.abs(xx))
            t2=np.argmax(np.abs(x))

            dt=0.002*(t2-t1)
        elif method==3:
            t1=np.argmax(np.abs(yy))
            t2=np.argmax(np.abs(y))
            dt=0.002*(t2-t1)
        elif method==4:
            t1=np.argmax(np.abs(zz))
            t2=np.argmax(np.abs(z))
            dt=0.002*(t2-t1)

        return dt
        
        
    def ShowShift(self,method):
        if self.hasoverlay == 1:
            try:
                t=self.filtered_data.loc[:,'time'].values
                x=self.filtered_data.loc[:,'x_ASTM'].values
                y=self.filtered_data.loc[:,'y_ASTM'].values
                z=self.filtered_data.loc[:,'z_ASTM'].values
            except:
                t=self.filtered_data.loc[:,'time'].values
                x=self.filtered_data.loc[:,'x_GB'].values
                y=self.filtered_data.loc[:,'y_GB'].values
                z=self.filtered_data.loc[:,'z_GB'].values
        
        
        a.clear()
        b.clear()
        c.clear()
        ao.clear()
        bo.clear()
        co.clear()
        a.plot(t,x,'b',linewidth=0.5)
        b.plot(t,y,'b',linewidth=0.5)
        c.plot(t,z,'b',linewidth=0.5)
        a.set_xlim(0)
        b.set_xlim(0)
        c.set_xlim(0)
        a.set_ylabel('Fore/Aft (g/s)')
        b.set_ylabel('Lateral (g/s)')
        c.set_ylabel('Vertical (g/s)')
        c.set_xlabel('time (s)')
        
        tt=self.filtered_overlay.loc[:,'time'].values
        xx=self.filtered_overlay.loc[:,'x'].values
        yy=self.filtered_overlay.loc[:,'y'].values
        zz=self.filtered_overlay.loc[:,'z'].values
        dt=self.MoveAxis(x,y,z,xx,yy,zz,method)
        ttt=tt+dt
        self.ShowOverlay(ttt,xx,yy,zz)     
        f.tight_layout()
        self.canvas.draw()
        
        
        
    def newtonviewer_helper(self):
        filename = filedialog.askopenfilename(filetypes=(('RawData', '*.txt'), ('All files', '*.*')))
        data=pd.read_table(filename)
        cols = data.shape[1]
        iter = int(cols/6)
        for i in range(iter):
            accdata=AccData(path=data.iloc[:,i*6:i*6+6],filefolder=filename)
            accdata.datasep(0)
            accdata.addtitle()
            fname='test iter'+str(i+1)
            accdata.writefile(fname)
        popupmsg('Done', 'Message')



    def RemoveOverlay(self):
        self.hasoverlay=False
        self.olbxmax.config(text=str())
        self.olbxmin.config(text=str())
        self.olbymax.config(text=str())
        self.olbymin.config(text=str())
        self.olbzmax.config(text=str())
        self.olbzmin.config(text=str()) 
        self.FilterShow(self.dataformat)


        
    def FilterOverlay(self):
        col=['time','x','y','z']
        self.filtered_overlay=pd.DataFrame(columns=col)
        for i in [0,1,2,3]:
            try:
                self.filtered_overlay[col[i]]=butter_lowpass_filter(self.overlaydata.rawdata.iloc[:,i].values, self.cutoff, 500, order=4)[249:]
            except:
                self.filtered_overlay[col[i]]=self.overlaydata.rawdata.iloc[:,i].values

                        
            
        tt=self.filtered_overlay.loc[:,'time'].values
        xx=self.filtered_overlay.loc[:,'x'].values
        yy=self.filtered_overlay.loc[:,'y'].values
        zz=self.filtered_overlay.loc[:,'z'].values
        #self.overlaynormstatus=0
        return tt,xx,yy,zz
        
        
        
    def ShowOverlay(self,tt,xx,yy,zz):

        m=ao.plot(tt,xx,'r',linewidth=0.5)
        n=bo.plot(tt,yy,'r',linewidth=0.5)
        op=co.plot(tt,zz,'r',linewidth=0.5)

        
        
        
    def Loadoverlay(self,method):
        
        if method=='file':
            self.overlaydata=OverlayData()
            self.overlayname=filedialog.askopenfilename(filetypes=(('OverlayData','*.txt'),('All files','*.*')))
            
            self.overlaydata.from_file(self.overlayname)
            self.overlaynormstatus=0
            self.hasoverlay=True
            self.FilterShow(self.dataformat)
        elif method=='lap':
            tkoverlay=tk.Toplevel()
            frame1=tk.Frame(tkoverlay)
            frame1.pack(side=tk.TOP,expand=False)
            label1=tk.Label(frame1,text='Enter Lap #: ')
            entry1=tk.Entry(frame1,width=4)
            list1=tk.Radiobutton(frame1,text='GB',value=1,command=lambda:self.RadioFunc('GB','overlay'))
            
            list2=tk.Radiobutton(frame1,text='ASTM',value=0,command=lambda:self.RadioFunc('ASTM','overlay'))
               
            list1.grid(row=0,column=0)
            list2.grid(row=0,column=1)
            label1.grid(row=1,column=0)
            entry1.grid(row=1,column=1)
            frame2=tk.Frame(tkoverlay)
            frame2.pack(side=tk.BOTTOM,expand=False)
            list1.deselect()
            list2.deselect()
            butt1=tk.Button(frame2,text='OK',command=lambda:self.loadoverlay_on_button(tkoverlay,entry1) )
            butt1.pack(anchor=tk.CENTER)
            self.overlaynormstatus=0
        
        

    
    def loadoverlay_on_button(self,tkoverlay,entry):
        self.overlaylap=int(entry.get())-1
        self.overlaydata=OverlayData()
        if self.overlayformat=='GB':
            self.overlaydata.from_lap(self.data.GBdata[self.overlaylap])
        else:
            self.overlaydata.from_lap(self.data.ASTMdata[self.overlaylap])
        self.hasoverlay=True
        self.FilterShow(self.dataformat)
        tkoverlay.destroy()
    
    
    def RadioFunc(self,method,obj):
        if obj == 'overlay':
    
            if method == 'GB':
                self.overlayformat='GB'
            else:
                self.overlayformat='ASTM'
        
        if obj == 'dataformat':
    
            if method == 'GB':
                self.dataformat='GB'
            else:
                self.dataformat='ASTM'
        
    

    
    
    def FilterShow(self,method):
        t,x,y,z=self.Filter(method)
        self.ShowPlot(t,x,y,z)
        
        self.overlaynormstatus=0
    
    def Filter(self,method):
        try:
            lap=int(self.etlaps.get())
        except:
            lap=0
        try:
            self.cutoff=float(self.etfilter.get())
        except:
            self.cutoff=500
        col=['time','x_GB','y_GB','z_GB','x_ASTM','y_ASTM','z_ASTM']
        self.filtered_data=pd.DataFrame(columns=col)
        colGB=['time','x_GB','y_GB','z_GB']

        if method == 'GB':
            self.Normstatus=0
            self.dataformat='GB'
            self.filtered_data=pd.DataFrame(columns=colGB)
            if lap>0:
                for i in range(4):
                    self.filtered_data[colGB[i]]=butter_lowpass_filter(self.data.GBdata[lap-1].loc[5:,colGB[i]], self.cutoff, 500, order=4)[249:]
            
            else:
                for i in range (4):
                    self.filtered_data[colGB[i]]=butter_lowpass_filter(self.data.rawdata.loc[:,colGB[i]], self.cutoff, 500, order=4)[249:]
               
            t=self.filtered_data.loc[:,'time'].values
            x=self.filtered_data.loc[:,'x_GB'].values
            y=self.filtered_data.loc[:,'y_GB'].values
            z=self.filtered_data.loc[:,'z_GB'].values
        if method == 'ASTM':
            self.Normstatus=0
            self.dataformat='ASTM'
            colASTM=['time','x_ASTM','y_ASTM','z_ASTM']
            self.filtered_data=pd.DataFrame(columns=colASTM)
            if lap>0:
                for i in range(4):
                    self.filtered_data[colASTM[i]]=butter_lowpass_filter(self.data.ASTMdata[lap-1].loc[5:,colASTM[i]], self.cutoff, 500, order=4)[249:]
                
            else:
                for i in range (4):
                    self.filtered_data[colASTM[i]]=butter_lowpass_filter(self.data.rawdata.loc[:,colASTM[i]], self.cutoff, 500, order=4)[249:]
                
        
            t=self.filtered_data.loc[:,'time'].values
            x=self.filtered_data.loc[:,'x_ASTM'].values
            y=self.filtered_data.loc[:,'y_ASTM'].values
            z=self.filtered_data.loc[:,'z_ASTM'].values
        return t,x,y,z
            
            
    def ShowPlot(self,t,x,y,z,overlaymethod=None):
        a.clear()
        b.clear()
        c.clear()
        ao.clear()
        bo.clear()
        co.clear()
        a.plot(t,x,'b',linewidth=0.5)
        b.plot(t,y,'b',linewidth=0.5)
        c.plot(t,z,'b',linewidth=0.5)
        a.set_xlim(0)
        b.set_xlim(0)
        c.set_xlim(0)
        a.set_ylabel('Fore/Aft (g/s)')
        b.set_ylabel('Lateral (g/s)')
        c.set_ylabel('Vertical (g/s)')
        c.set_xlabel('time (s)')
        if self.hasoverlay is True:
            if overlaymethod=='Norm':
                tt,xx,yy,zz=self.NormalizationOverlay()
                self.ShowOverlay(tt,xx,yy,zz)
            elif overlaymethod=='UnNorm':
                tt,xx,yy,zz=self.UnNormalizationOverlay()
                self.ShowOverlay(tt,xx,yy,zz)                
            else:
                tt,xx,yy,zz=self.FilterOverlay()
                self.ShowOverlay(tt,xx,yy,zz)

        
        
        
        
        
        
        f.tight_layout()
        self.canvas.draw()
        self.status.config(text='Data filtered with LPF: ' +str(self.cutoff) +'Hz')
        if self.hasoverlay:
            self.path.config(text='Current File on Display: '+self.filename+'\nOverlay File name: '+self.overlaydata.filename)
        else:
            self.path.config(text='Current File on Display: ' + self.filename)
        self.lbxmax.config(text=str(np.round(np.max(x),4)))
        self.lbxmin.config(text=str(np.round(np.min(x),4)))
        self.lbymax.config(text=str(np.round(np.max(y),4)))
        self.lbymin.config(text=str(np.round(np.min(y),4)))
        self.lbzmax.config(text=str(np.round(np.max(z),4)))
        self.lbzmin.config(text=str(np.round(np.min(z),4)))
        self.olbxmax.config(text=str(np.round(np.max(xx),4)))
        self.olbxmin.config(text=str(np.round(np.min(xx),4)))
        self.olbymax.config(text=str(np.round(np.max(yy),4)))
        self.olbymin.config(text=str(np.round(np.min(yy),4)))
        self.olbzmax.config(text=str(np.round(np.max(zz),4)))
        self.olbzmin.config(text=str(np.round(np.min(zz),4)))        
        #self.filtered_data.head()
    
    
    def Normalization(self):
        if self.Normstatus==0:
            self.Normstatus=1
            try:
                self.filtered_data.loc[:,['x_GB','y_GB','z_GB']]-=self.data.offset[0:3]
                t=self.filtered_data.loc[:,'time'].values
                x=self.filtered_data.loc[:,'x_GB'].values
                y=self.filtered_data.loc[:,'y_GB'].values
                z=self.filtered_data.loc[:,'z_GB'].values
            except:
                self.filtered_data.loc[:,['x_ASTM','y_ASTM','z_ASTM']]-=self.data.offset[3:6] 
                t=self.filtered_data.loc[:,'time'].values
                x=self.filtered_data.loc[:,'x_ASTM'].values
                y=self.filtered_data.loc[:,'y_ASTM'].values
                z=self.filtered_data.loc[:,'z_ASTM'].values
        return t,x,y,z
            
            
    def UnNormalization(self):

        if self.Normstatus==1:
            self.Normstatus=0
            try:
                self.filtered_data.loc[:,['x_GB','y_GB','z_GB']]+=self.data.offset[0:3]
                t=self.filtered_data.loc[:,'time'].values
                x=self.filtered_data.loc[:,'x_GB'].values
                y=self.filtered_data.loc[:,'y_GB'].values
                z=self.filtered_data.loc[:,'z_GB'].values
            except:
                self.filtered_data.loc[:,['x_ASTM','y_ASTM','z_ASTM']]+=self.data.offset[3:6] 
                t=self.filtered_data.loc[:,'time'].values
                x=self.filtered_data.loc[:,'x_ASTM'].values
                y=self.filtered_data.loc[:,'y_ASTM'].values
                z=self.filtered_data.loc[:,'z_ASTM'].values
        return t,x,y,z
        
    def NormalizationOverlay(self):
        if self.overlaynormstatus==0:
            self.overlaydata.get_offset()
            self.overlaynormstatus=1
            self.filtered_overlay.loc[:,['x','y','z']]-=self.overlaydata.offset[1:] 
            tt=self.filtered_overlay.loc[:,'time'].values
            xx=self.filtered_overlay.loc[:,'x'].values
            yy=self.filtered_overlay.loc[:,'y'].values
            zz=self.filtered_overlay.loc[:,'z'].values

            return tt,xx,yy,zz
            
            
    def UnNormalizationOverlay(self):
        if self.overlaynormstatus==1:
            self.overlaynormstatus=0
            self.filtered_overlay.loc[:,['x','y','z']]+=self.overlaydata.offset[1:] 
            tt=self.filtered_overlay.loc[:,'time'].values
            xx=self.filtered_overlay.loc[:,'x'].values
            yy=self.filtered_overlay.loc[:,'y'].values
            zz=self.filtered_overlay.loc[:,'z'].values

            return tt,xx,yy,zz
       
        
                
    def ShowNorm(self):
        t,x,y,z=self.Normalization()
        self.ShowPlot(t,x,y,z,'Norm')
        
        
    def ShowUnNorm(self,overlaymethod=None):
        t,x,y,z=self.UnNormalization()
        self.ShowPlot(t,x,y,z,'UnNorm')
        
            
        
        
        
    def Openfile(self):
        self.filename=filedialog.askopenfilename(filetypes=(('AccData','*.txt'),('All files','*.*')))
        self.data=AccData(self.filename)
    
    def Preview(self,method=None):
        self.Normstatus=0
        if method=='reset':
            self.hasoverlay=False
        if self.dataformat == 'GB':
            t=self.data.rawdata.loc[:,'time'].values
            x=self.data.rawdata.loc[:,'x_GB'].values
            y=self.data.rawdata.loc[:,'y_GB'].values
            z=self.data.rawdata.loc[:,'z_GB'].values
            self.filtered_data=pd.DataFrame(np.transpose([t,x,y,z]),columns=['time','x_GB','y_GB','z_GB'])
        elif self.dataformat == 'ASTM':
            t=self.data.rawdata.loc[:,'time'].values
            x=self.data.rawdata.loc[:,'x_ASTM'].values
            y=self.data.rawdata.loc[:,'y_ASTM'].values
            z=self.data.rawdata.loc[:,'z_ASTM'].values
            self.filtered_data=pd.DataFrame(np.transpose([t,x,y,z]),columns=['time','x_ASTM','y_ASTM','z_ASTM'])
        self.status.config(text='Preview: ASTM Data.')
        self.path.config(text='Current File on Display: '+self.filename)
        self.ShowPlot(t,x,y,z)


        
    def OpenandPreview(self):  
    
        self.Openfile()
        self.Preview()

        
    def DataCut(self):
        self.Top=tk.Toplevel()
        self.Top.title='Save File'
        label1=tk.Label(self.Top,text='Number of Laps: ')
        label2=tk.Label(self.Top,text='Start Time of Each Lap: ')
        self.entry1=tk.Entry(self.Top)
        self.entry2=tk.Entry(self.Top)
        button1=ttk.Button(self.Top, text='Separate Data',command=lambda: self.SepData())
        label1.grid(row=0,column=0)
        self.entry1.grid(row=0,column=1)
        label2.grid(row=1,column=0)
        self.entry2.grid(row=1,column=1)
        label3=tk.Label(self.Top,text='Enter File Name to Save: ')
        label3.grid(row=2,column=0)
        self.entry3=tk.Entry(self.Top)
        self.entry3.grid(row=2,column=1)
        button1.grid(row=3,columnspan=2,sticky='nswe')
        
        
    def SepData(self):
        nlaps=int(self.entry1.get())
        tlaps=np.fromstring(self.entry2.get(),dtype=float, sep=',')
        self.fname=self.entry3.get()
        self.data.changelaps(nlaps)
        self.data.datasep(tlaps)
        self.data.addtitle()
        self.status.config(text='Data separated successfully!')
        self.Top.destroy()

    def SaveData(self):
        self.data.writefile(self.fname)
        self.status.config(text='Data exported successfully!')

        
    def GenerateGB(self):        
        try:
            self.GB.destroy()

        except:
            pass
        

        data10=self.filtered_data
        d0py=max(0,np.max(data10.iloc[:,2].values))
        d0ny=min(0,np.min(data10.iloc[:,2].values))
        d0pz=max(1,np.max(data10.iloc[:,3].values))
        d0nz=min(1,np.min(data10.iloc[:,3].values))
        d1p,d1n=ProcessGB(data10,5)
        d2p,d2n=ProcessGB(data10,25)
        d3p,d3n=ProcessGB(data10,100)
        d4p,d4n=ProcessGB(data10,250)
        d5p,d5n=ProcessGB(data10,500)
        d6p,d6n=ProcessGB(data10,1000)
        d7p,d7n=ProcessGB(data10,2000)
        z_array=self.filtered_data.iloc[:,3].values
        y_array=np.abs(self.filtered_data.iloc[:,2].values)
        p1=[d0py,d1p[0],d2p[0],d3p[0],d4p[0],d5p[0],d6p[0],d7p[0]]
        p2=[-d0ny,-d1n[0],-d2n[0],-d3n[0],-d4n[0],-d5n[0],-d6n[0],-d7n[0]]
        p3=[d0pz,d1p[1],d2p[1],d3p[1],d4p[1],d5p[1],d6p[1],d7p[1]]
        p4=[d0nz,d1n[1],d2n[1],d3n[1],d4n[1],d5n[1],d6n[1],d7n[1]]
        initializeF2()
        aa.loglog([0.002,0.01,0.05,0.2,0.5,1,2,4],p1,'b',label='Measured ay(+)')
        aa.loglog([0.002,0.01,0.05,0.2,0.5,1,2,4],p2,'k',label='Measured ay( - )')
        bb.plot([0.002,0.01,0.05,0.2,0.5,1,2,4],p3,'b',label='Measured az(+)')
        bb.plot([0.002,0.01,0.05,0.2,0.5,1,2,4],p4,'k',label='Measured az( - )')
        dd.plot(z_array,y_array,'x-',color='b',label='Measured Data')
        aa.legend()
        bb.legend()
        dd.legend()        
        self.PlotGB()
        
    def PlotGB(self):
        
        self.GB=tk.Toplevel(bg='LightBlue1')
        self.GB.wm_title('GB Standard')
        frame=tk.Frame(self.GB,bg='LightBlue1')
        frame.pack(side=tk.TOP,anchor='e',expand=True)
        frame1=tk.Frame(frame,bg='LightBlue1')
        frame1.pack(side=tk.RIGHT)
        label1=tk.Label(frame1,bg='LightBlue1',text='Measured Max/Allowable Max ax: ')
        label2=tk.Label(frame1,bg='LightBlue1',text='Measured Min/Allowable Min ax: ')
        label3=tk.Label(frame1,text=str(np.round(np.max(self.filtered_data.iloc[:,1].values),4))+' / 6',fg='green',bg='LightBlue1')
        label4=tk.Label(frame1,text=str(np.round(np.min(self.filtered_data.iloc[:,1].values),4))+' / -3.5',fg='green',bg='LightBlue1')
        if np.max(self.filtered_data.iloc[:,1].values)>6 or np.min(self.filtered_data.iloc[:,1].values)<-3.5:
            label3.config(fg='red')
            label4.config(fg='red')
        label1.grid(row=0,column=0,sticky='nswe')
        label2.grid(row=1,column=0,sticky='nswe')
        label3.grid(row=0,column=1,sticky='nswe')
        label4.grid(row=1,column=1,sticky='nswe')
        self.GB.wm_title('GB Standard')
        canvas2 = FigureCanvasTkAgg(f2, self.GB)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.status.config(text='Report Generated.')
        
        toolbar = NavigationToolbar2Tk(canvas2, self.GB)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.BOTTOM, expand=True)
        
        
        
    def GenerateASTM(self):        
        try:
            self.ASTM.destroy()

        except:
            pass
        data10=self.filtered_data
        array=data10.iloc[:,1:4].values
        d0p,d0n=ProcessASTM(data10,5)
        
        d1p,d1n=ProcessASTM(data10,10)
        d2p,d2n=ProcessASTM(data10,25)
        d3p,d3n=ProcessASTM(data10,50)
        d4p,d4n=ProcessASTM(data10,250)
        d5p,d5n=ProcessASTM(data10,500)
        d6p,d6n=ProcessASTM(data10,2000)
        d7p,d7n=ProcessASTM(data10,100)
        
        d8p,d8n=ProcessASTM(data10,7000)
        d9p,d9n=ProcessASTM(data10,1000)
        d10p,d10n=ProcessASTM(data10,2500)
        d77p,d77n=ProcessASTM(data10,3000)
        d99p,d99n=ProcessASTM(data10,3500)
        d111p,d111n=ProcessASTM(data10,4500)
        
        
        initializeF3()
        a3.plot([0.002,0.02,0.05,0.1,0.2,0.5,1,2,4,5,6,7,9,14],[d0p[0],d1p[0],d2p[0],d3p[0],d7p[0],d4p[0],d5p[0],d9p[0],d6p[0],d10p[0],d77p[0],d99p[0],d111p[0],d8p[0]],'b',label='Measured ax (+)')
        a3.plot([0.002,0.02,0.05,0.1,0.2,0.5,1,2,4,5,6,7,9,14],[d0n[0],d1n[0],d2n[0],d3n[0],d7n[0],d4n[0],d5n[0],d9n[0],d6n[0],d10n[0],d77n[0],d99n[0],d111n[0],d8n[0]],'green',label='Measured ax (-)')
        b3.plot([0.002,0.02,0.05,0.1,0.2,0.5,1,2,4,5,6,7,9,14],[d0p[1],d1p[1],d2p[1],d3p[1],d7p[1],d4p[1],d5p[1],d9p[1],d6p[1],d10p[1],d77p[1],d99p[1],d111p[1],d8p[1]],'b',label='Measured ay (Right)')
        b3.plot([0.002,0.02,0.05,0.1,0.2,0.5,1,2,4,5,6,7,9,14],[-d0n[1],-d1n[1],-d2n[1],-d3n[1],-d7n[1],-d4n[1],-d5n[1],-d9n[1],-d6n[1],-d10n[1],-d77n[1],-d99n[1],-d111n[1],-d8n[1]],'green',label='Measured ay (Left)')
        c3.plot([0.002,0.02,0.05,0.1,0.2,0.5,1,2,4,5,6,7,9,14],[d0p[2],d1p[2],d2p[2],d3p[2],d7p[2],d4p[2],d5p[2],d9p[2],d6p[2],d10p[2],d77p[2],d99p[2],d111p[2],d8p[2]],'b',label='Measured az (Down)')
        c3.plot([0.002,0.02,0.05,0.1,0.2,0.5,1,2,4,5,6,7,9,14],[d0n[2],d1n[2],d2n[2],d3n[2],d7n[2],d4n[2],d5n[2],d9n[2],d6n[2],d10n[2],d77n[2],d99n[2],d111n[2],d8n[2]],'green',label='Measured az ( Up )')
        a3.legend()
        b3.legend()
        c3.legend()
        d3.plot(array[:,0],array[:,1],'bx')
        e3.plot(array[:,0],array[:,2],'bx')
        ff3.plot(array[:,1],array[:,2],'bx')
        self.PlotASTM()
        
    def PlotASTM(self):
        
        self.ASTM=tk.Toplevel(bg='LightBlue1')
        self.ASTM.geometry('1200x800')
        self.ASTM.wm_title('ASTM Standard')
        menu=tk.Frame(self.ASTM,bg='LightBlue1')
        menu.pack(side=tk.TOP,anchor='n',expand=True)
        lb1=tk.Label(menu,text='Select Restraint Type: ',bg='LightBlue1')
        lb1.grid(row=0,column=0,sticky='we')
        lb2=tk.Label(menu,text='Patron Height (Inches): ',bg='LightBlue1')
        lb2.grid(row=0,column=1,sticky='we')
        height=tk.Entry(menu)
        height.grid(row=1,column=1,sticky='we')        
        v=['Individual Lower Body','Upper Body','Group Lower Body','No Restraint','Convenience Restraint','Remove Restraint Type']
        v2=['Normal Run','E-Stop']
        self.combo=ttk.Combobox(menu,values=v,width=23)
        self.combo.grid(row=1,column=0,sticky='we')
        self.combo2=ttk.Combobox(menu,values=v2,width=23)
        self.combo2.grid(row=2,column=0,sticky='we')
        button=ttk.Button(menu,text='Update',command=lambda:self.DisneyStd(canvas3,height))
        button.grid(row=1,column=3,rowspan=2,sticky='nse')


        canvas3 = FigureCanvasTkAgg(f3, self.ASTM)
        canvas3.draw()
        canvas3.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.status.config(text='Report Generated.')
        
        toolbar = NavigationToolbar2Tk(canvas3, self.ASTM)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.BOTTOM, expand=True)
        self.ASTM.mainloop()
    
    def Runtype(self):
        runtp=self.combo2.get()
        if runtp=='E-Stop':
            return 1.25
        else:
            return 1
    def DisneyStd(self,canvas,height):
        restype=self.combo.get()
        try:
            ht=float(height.get())
        except:
            ht=1
        try:
            runtype=self.Runtype()
        except:
            runtype=1
        
        
        try:
            del ff3.lines[4:]
            del e3.lines[4:]
            del d3.lines[4:]
            del c3.lines[4:]
            del b3.lines[3:]
            del a3.lines[4:]
        except:
            pass

        if restype == 'Upper Body':
            
            xa,ya=eggXY(2*runtype*coef(ht,'x'),3.6*runtype*coef(ht,'x'),3*runtype*coef(ht,'x'))
            xaa,yaa=eggXY(1.6*runtype*coef(ht,'x'),3.6*runtype*coef(ht,'x'),2.4*runtype*coef(ht,'x'))
            xb,yb=eggXZ(2*runtype*coef(ht,'z'),3.6*runtype*coef(ht,'z'),2*runtype*coef(ht,'z'),5*runtype*coef(ht,'z'))
            xbb,ybb=eggXZ(1.6*runtype*coef(ht,'z'),3.6*runtype*coef(ht,'z'),1.4*runtype*coef(ht,'z'),4.8*runtype*coef(ht,'z'))
            xc,yc=eggYZ(3*runtype*coef(ht,'z'),2*runtype*coef(ht,'z'),5*runtype*coef(ht,'z'))
            xcc,ycc=eggYZ(2.4*runtype*coef(ht,'z'),1.4*runtype*coef(ht,'z'),4.8*runtype*coef(ht,'z'))
            
            d3.plot(xa,ya,'k',label='Upper Body (0s)')
            d3.plot(xaa,yaa,'k--',label='Upper Body (0.2s)')
            e3.plot(xb,yb,'k',label='Upper Body (0s)')
            e3.plot(xbb,ybb,'k--',label='Upper Body (0.2s)')            
            ff3.plot(xc,yc,'k',label='Upper Body (0s)')
            ff3.plot(xcc,ycc,'k--',label='Upper Body (0.2s)')            
                        
            PlotDuration(a3,b3,c3,runtype*coef(ht,'x'),runtype*coef(ht,'z'),restype)
            
            
            
            

        
        elif restype == 'Group Lower Body':

            xa,ya=eggXY(1.7*runtype*coef(ht,'x'),2.5*runtype*coef(ht,'x'),2.4*runtype*coef(ht,'x'))
            xaa,yaa=eggXY(1.4*runtype*coef(ht,'x'),2.5*runtype*coef(ht,'x'),2.1*runtype*coef(ht,'x'))
            xb,yb=eggXZ(1.7*runtype*coef(ht,'z'),2.5*runtype*coef(ht,'z'),2*runtype*coef(ht,'z'),3.5*runtype*coef(ht,'z'))
            xbb,ybb=eggXZ(1.4*runtype*coef(ht,'z'),2.5*runtype*coef(ht,'z'),1*runtype*coef(ht,'z'),3*runtype*coef(ht,'z'))
            xc,yc=eggYZ(2.4*runtype*coef(ht,'z'),2*runtype*coef(ht,'z'),3.5*runtype*coef(ht,'z'))
            xcc,ycc=eggYZ(2.1*runtype*coef(ht,'z'),1*runtype*coef(ht,'z'),3*runtype*coef(ht,'z'))
            
            d3.plot(xa,ya,'k',label='Group Lower Body (0s)')
            d3.plot(xaa,yaa,'k--',label='Group Lower Body (0.2s)')
            e3.plot(xb,1+yb,'k',label='Group Lower Body (0s)')
            e3.plot(xbb,1+ybb,'k--',label='Group Lower Body (0.2s)')            
            ff3.plot(xc,1+yc,'k',label='Group Lower Body (0s)')
            ff3.plot(xcc,1+ycc,'k--',label='Group Lower Body (0.2s)')  
            
            
            
            PlotDuration(a3,b3,c3,runtype*coef(ht,'x'),runtype*coef(ht,'z'),restype)
        elif restype == 'Individual Lower Body':
            
            xa,ya=eggXY(1.8*runtype*coef(ht,'x'),2.5*runtype*coef(ht,'x'),2.6*runtype*coef(ht,'x'))
            xaa,yaa=eggXY(1.5*runtype*coef(ht,'x'),2.5*runtype*coef(ht,'x'),2.2*runtype*coef(ht,'x'))
            xb,yb=eggXZ(1.8*runtype*coef(ht,'z'),2.5*runtype*coef(ht,'z'),1.8*runtype*coef(ht,'z'),4.8*runtype*coef(ht,'z'))
            xbb,ybb=eggXZ(1.5*runtype*coef(ht,'z'),2.5*runtype*coef(ht,'z'),1.2*runtype*coef(ht,'z'),4.5*runtype*coef(ht,'z'))
            xc,yc=eggYZ(2.6*runtype*coef(ht,'z'),1.8*runtype*coef(ht,'z'),4.8*runtype*coef(ht,'z'))
            xcc,ycc=eggYZ(2.2*runtype*coef(ht,'z'),1.2*runtype*coef(ht,'z'),4.5*runtype*coef(ht,'z'))
            
            d3.plot(xa,ya,'k',label='Individual Lower Body (0s)')
            d3.plot(xaa,yaa,'k--',label='Individual Lower Body (0.2s)')
            e3.plot(xb,yb,'k',label='Individual Lower Body (0s)')
            e3.plot(xbb,ybb,'k--',label='Individual Lower Body (0.2s)')            
            ff3.plot(xc,yc,'k',label='Individual Lower Body (0s)')
            ff3.plot(xcc,ycc,'k--',label='Individual Lower Body (0.2s)')
            

            PlotDuration(a3,b3,c3,runtype*coef(ht,'x'),runtype*coef(ht,'z'),restype)
            
            
        elif restype == 'No Restraint' or restype=='Convenience Restraint':
            
            xa,ya=eggXY(1.5*runtype*coef(ht,'x'),2.5*runtype*coef(ht,'x'),1.8*runtype*coef(ht,'x'))
            xaa,yaa=eggXY(1.2*runtype*coef(ht,'x'),2.5*runtype*coef(ht,'x'),1.2*runtype*coef(ht,'x'))
            xb,yb=eggXZ(1.5*runtype*coef(ht,'z'),2.5*runtype*coef(ht,'z'),1.2*runtype*coef(ht,'z'),3*runtype*coef(ht,'z'))
            xbb,ybb=eggXZ(1.2*runtype*coef(ht,'z'),2.5*runtype*coef(ht,'z'),0.8*runtype*coef(ht,'z'),2.8*runtype*coef(ht,'z'))
            xc,yc=eggYZ(1.8*runtype*coef(ht,'z'),1.2*runtype*coef(ht,'z'),3*runtype*coef(ht,'z'))
            xcc,ycc=eggYZ(1.2*runtype*coef(ht,'z'),0.8*runtype*coef(ht,'z'),2.8*runtype*coef(ht,'z'))
            
            d3.plot(xa,ya,'k',label='No/Conv Restraint (0s)')
            d3.plot(xaa,yaa,'k--',label='No/Conv Restraint (0.2s)')
            e3.plot(xb,1+yb,'k',label='No/Conv Restraint (0s)')
            e3.plot(xbb,1+ybb,'k--',label='No/Conv Restraint (0.2s)')            
            ff3.plot(xc,1+yc,'k',label='No/Conv Restraint (0s)')
            ff3.plot(xcc,1+ycc,'k--',label='No/Conv Restraint (0.2s)')      
            
            
            PlotDuration(a3,b3,c3,runtype*coef(ht,'x'),runtype*coef(ht,'z'),restype)
            
        elif restype == 'Remove Restraint Type':
            pass
        
                        
        a3.legend()
        b3.legend()
        c3.legend()
        d3.legend()
        e3.legend()
        ff3.legend()
        canvas.draw_idle()        
        
main=gui()
main.mainloop()