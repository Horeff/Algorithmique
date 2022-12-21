from matplotlib import pyplot as plt
import numpy as np
import types

class figures():
    def __init__(self) -> None:
        pass

    def segment(self, c : tuple, l : float = 1, arg : float = 0, expr_x = None, expr_y = None, args : list = None):
        if expr_x is None and expr_y is None:
            x = [c[0], l]
            y = [c[1], 0]
            x_res,y_res = self.rotation(x, y, arg)
        else:
            lsx = np.linspace(0, l, 10*l)
            rajx = c[0]
            rajy = c[1]
            x_res = []
            y_res = []
            if isinstance(expr_y, types.FunctionType):
                for i in range(len(lsx)):
                    x_res.append(lsx[i])
                    y_res.append(expr_y(lsx[i]))
            else:
                if args is not None:
                    x_res,y_res = expr_y((0,0),args = args,l = l)
                else:
                    x_res,y_res = expr_y((0,0),l = l)
            x_res,y_res = self.rotation(x_res, y_res, arg)
            for i in range(len(x_res)):
                x_res[i] += rajx
                y_res[i] += rajy
        x_res = np.insert(x_res, 0, rajx)
        y_res = [rajy]+y_res
        return x_res,y_res

    def rotation(self,x,y,arg):
        rajx = x[0]
        rajy = y[0]
        for i in range(len(x)):
            x[i] = x[i] - rajx
            y[i] = y[i] - rajy
        for i in range(len(x)):
            if x[i] != 0:
                mod = np.sqrt(x[i]**2+y[i]**2)
                alpha = np.arctan(y[i]/x[i])
            elif y[i] != 0:
                mod = y[i]
                alpha = np.pi/2
            try:
                x[i] = mod*np.cos(arg+alpha)
                y[i] = mod*np.sin(arg+alpha)
            except:pass
        for i in range(len(x)):
            x[i] = x[i] + rajx
            y[i] = y[i] + rajy
        return x,y

    def carre(self, co : tuple, args : list = None, c : int = 4, l : float = 1, cot : float = 1, prop : bool = 1) -> list :
        if args is not None:
            try:
                c = args[0]
            except:pass
            try:
                s = args[1]
            except:pass
            try:
                prop = args[2]
            except:pass
        if prop:
            p = [0,0,1,1,0]
            s = [0,1,1,0,0]
        else:
            p = [0,0,1,1,0]
            s = [0,-1,-1,0,0]
        x_res = [co[0]]
        y_res = [co[1]]
        for i in range(l):
            x_res += [x_res[-1]+p[i]*cot for i in range(c+1)]
            y_res += [y_res[-1]+s[i]*cot for i in range(c+1)]
        return x_res,y_res

    def rectangle(self, co : tuple, args : list = None, c : int = 4, l : float = 1, cot1 : float = 2, cot : float = 1, prop : bool = 1) -> list :
        if args is not None:
            try:
                c = args[0]
            except:pass
            try:
                s = args[1]
            except:pass
            try:
                prop = args[2]
            except:pass
        if prop:
            p = [0,0,1,1,0]
            s = [0,1,1,0,0]
        else:
            p = [0,0,1,1,0]
            s = [0,-1,-1,0,0]
        x_res = [co[0]]
        y_res = [co[1]]
        for i in range(l):
            x_res += [x_res[-1]+p[i]*cot for i in range(c+1)]
            y_res += [y_res[-1]+s[i]*cot1 for i in range(c+1)]
        return x_res,y_res

    def expr_def(self, x):
        return 0

    def circle(self, c : tuple, a : float = np.pi, r : float = 1, t : float = 0, expr = None) -> list :
        ls = np.linspace(0,a,1000)
        if expr is None:
            expr = self.expr_def
        x = [(expr(i)+r)*np.cos(i+t)+c[0] for i in ls]
        y = [(expr(i)+r)*np.sin(i+t)+c[1] for i in ls]
        return x,y

class fonctions():
    def __init__(self) -> None:
        pass

    def sinus(self,x,y,l):
        ls = np.linspace(x,x+l, 10*l)
        return ls,[np.sin(i) for i in ls]
    
    def ech(self,x : float):
        if int(x)%2 == 0:
            return True
        else:
            return False
    
    def echelon_per(self, c : tuple, args : list = None, r : float = 100, l : float = 10, p1 : float = 1, p2 : float = -1, expr = None):
        if args is not None:
            try:
                p1 = args[0]
            except: pass
            try:
                p2 = args[1]
            except: pass
            try:
                expr = args[2]
            except: pass
            try:
                r = args[3]
            except:pass
        ls = np.linspace(c[0],c[0]+l,r*l)
        y_res = []
        if expr is None:
            expr = self.ech
        for i in ls:
            if expr(i):
                y_res.append(c[1]+p1)
            else:
                y_res.append(c[1]+p2)
        return ls,y_res

class bibliotheque():
    def __init__(self) -> None:
        self.fig = figures()
        self.func = fonctions()
    
    def carre_esca(self,c : tuple, n : int = 4, arg : float = 0, args : list = None, expr = None, argraj : list = None):
        if args is None:
            args = [[2,1,1],[2,1,0],[2,1,1],[2,1,0]]
            expr = self.fig.carre
            argraj = [0,0,np.pi,np.pi]
        x = []
        y = []
        x,y = self.fig.segment((c[0],c[1]), l = n, arg = arg+argraj[0], expr_y = expr, args = args[0])
        x_p,y_p = self.fig.segment((x[-1],y[-1]), l = n-1, arg = arg+argraj[1], expr_y = expr, args = args[1])
        x = list(x) + list(x_p)
        y += y_p
        x_p,y_p = self.fig.segment((x[-1],y[-1]), l = n, arg = arg+argraj[2], expr_y = expr, args = args[2])
        x = list(x) + list(x_p)
        y += y_p
        x_p,y_p = self.fig.segment((x[-1],y[-1]), l = n-1, arg = arg+argraj[3], expr_y = expr, args = args[3])
        x = list(x) + list(x_p)
        y += y_p
        return x,y

    def cercle_vaguelettes(self,c : tuple, r : float = 1, v : float = 20, a : float = 10, expr = None):
        def sinus(x):
            return np.cos(v*x)*r/a
        if expr is None:
            expr = sinus
        x,y = fig.circle((c[0],c[1]),a=2*np.pi,expr=expr, r = r)
        return x,y
    
    def carre_ds_carre(self, c : tuple, r : float = 1, n : int = 10, arg : float = np.pi/18):
        x_res,y_res = self.fig.carre(c,c = 4, cot = r)
        cot = r
        coord = [c[0],c[1]]
        for i in range(n):
            y_p = (np.tan(arg)*cot)/(np.tan(arg)+1)
            coord = [coord[0] + y_p*np.cos(i*arg), coord[1] + y_p*np.sin(i*arg)]
            cot = y_p/np.sin(arg)
            x,y = self.fig.carre(coord,c = 4, cot = cot)
            x,y = self.fig.rotation(x,y,(i+1)*arg)
            x_res += x + [coord[0]]
            y_res += y + [coord[1]]
        return x_res,y_res
    
    def fenetre(self,c : tuple,l : float = 1, h : float = 1.5):
        x,y = self.fig.rectangle(c,cot = l, cot1 = abs(h-l))
        x_p,y_p = self.fig.circle((x[-3]-l/2,y[-3]), r = abs(l)/2)
        return x+[c[0]]+x_p,y+[c[1]+abs(h-l)]+y_p
    
    def chateau(self,c : tuple):
        x_res,y_res = [c[0]],[c[1]]
        x,y = self.fig.rectangle((10,10), cot = 400, cot1 = 600)
        x_res += x
        y_res += y
        x,y = self.fenetre((200,300),20,101)
        plt.fill(x,y,color='black')
        plt.plot(x,y,color = "black")
        return x_res,y_res



fig = figures()
func = fonctions()
bib = bibliotheque()
#x,y = fig.segment((10,10), l = 100, arg = np.pi/4)
#x,y = func.sinus(0,0,50)
#x,y = fig.carre((0,0),3)
#x,y = fig.rotation(x,y,np.pi/4)
#x,y = fig.circle((0,0), a = 2*np.pi/3, t = np.pi/6)
#x,y = fig.segment((0,0), l = 20, arg = np.pi/4, args = [2,-2], expr_y=func.echelon_per)
#x,y = bib.carre_esca((0,0), n = 10, args = [[-1,1],[-1,1],[-1,1],[-1,1]], expr = func.echelon_per, argraj = [np.pi/4, -np.pi/4, -3*np.pi/4, 3*np.pi/4])
#x,y = bib.carre_esca((0,0), n = 10)
#x,y = bib.cercle_vaguelettes((0,0),r = 2, v = 20, a = 10)
#x,y = bib.carre_ds_carre((0,0))
#x,y = bib.chateau((0,0))
#plt.axis("equal")
#plt.plot(x,y)
#plt.show()
