import cv2
import numpy as np
import sys

def resize(*args, **kwargs):
    return cv2.resize(*args, **kwargs)

def moveWindow(*args,**kwargs):
    return

def imshow(*args,**kwargs):
    return cv2.imshow(*args,**kwargs)
    
def destroyWindow(*args,**kwargs):
    return cv2.destroyWindow(*args,**kwargs)

def waitKey(*args,**kwargs):
    return cv2.waitKey(*args,**kwargs)

def combine(left, right):
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    hoff = left.shape[0]
    
    shape = list(left.shape)
    shape[0] = h
    shape[1] = w
    
    comb = np.zeros(tuple(shape),left.dtype)
    
    comb[:left.shape[0],:left.shape[1]] = left
    comb[:right.shape[0],left.shape[1]:] = right
    
    return comb   


def peakdet(v, delta, x = None):
    maxtab = []
    mintab = []
           
    if x is None:
        x = np.arange(len(v))
        
    v = np.asarray(v)
       
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
        
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
        
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
       
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
        
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
            
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)            
    
def plotXY(data,size = (480,640),margin = 25,name = "data",labels=[], skip = [],
           showmax = [], bg = None,label_ndigits = [], showmax_digits=[]):
    
    mix = []
    maxtab, mintab = peakdet(data[0][1], 0.3)
    if(len(maxtab)>0 and len(mintab)>0):
        mix = np.append(maxtab[...,0],mintab[...,0])
        mix = np.sort(mix)
        mix = mix.astype(int)

    for x,y in data:
        if len(x) < 2 or len(y) < 2:
            return
    
    n_plots = len(data)
    w = float(size[1])
    h = size[0]/float(n_plots)
    
    z = np.zeros((size[0],size[1],3))
    
    if isinstance(bg,np.ndarray):
        wd = int(bg.shape[1]/bg.shape[0]*h )
        bg = cv2.resize(bg,(wd,int(h)))
        if len(bg.shape) == 3:
            r = combine(bg[:,:,0],z[:,:,0])
            g = combine(bg[:,:,1],z[:,:,1])
            b = combine(bg[:,:,2],z[:,:,2])
        else:
            r = combine(bg,z[:,:,0])
            g = combine(bg,z[:,:,1])
            b = combine(bg,z[:,:,2])
        z = cv2.merge([r,g,b])[:,:-wd,]    
    
    i = 0
    P = []
    for x,y in data:
        x = np.array(x)
        y = -np.array(y)
        
        xx = (w-2*margin)*(x - x.min()) / (x.max() - x.min())+margin
        yy = (h-2*margin)*(y - y.min()) / (y.max() - y.min())+margin + i*h
        mx = max(yy)
        if labels:
            if labels[i]:
                for ii in range(len(x)):
                    if ii%skip[i] == 0:
                        col = (255,255,255)
                        col2 = (255,0,0)
                        ss = '{0:.%sf}' % label_ndigits[i]
                        ss = ss.format(x[ii])
                        cv2.putText(z,ss,(int(xx[ii]),int((i+1)*h)),
                                    cv2.FONT_HERSHEY_PLAIN,1,col)           
        if showmax:
            if showmax[i]:
                col = (0,255,0)    
                ii = np.argmax(-y)
                ss = '{0:.%sf} %s' % (showmax_digits[i], showmax[i])
                ss = ss.format(x[ii]) 
                cv2.putText(z,ss,(int(xx[ii]),int((yy[ii]))),
                            cv2.FONT_HERSHEY_PLAIN,2,col)
       
        try:
            pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
            i+=1
            P.append(pts)
        except ValueError:
            pass 

    for p in P:
        m = []
        for i in range(len(p)-1):
            cv2.line(z,tuple(p[i]),tuple(p[i+1]), (255,255,255),1)           
    cv2.imshow(name,z)
    