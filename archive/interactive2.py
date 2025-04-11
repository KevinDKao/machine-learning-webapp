from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class knn_1d_reg(object):
    def __init__(self):
        super().__init__()
        np.random.seed(0)
        self.n=1000
        self.xlims = [0, 2*np.pi]
        self.xlims_range = self.xlims[1] - self.xlims[0]
        self.xlims_ext = [self.xlims[0]-.1*self.xlims_range, self.xlims[1]+.1*self.xlims_range]
        # self.x = np.linspace(self.xlims[0], self.xlims[1],self.n)
        self.x = (np.random.randn(self.n)/6 + 0.5)*(self.xlims[1]-self.xlims[0]) + self.xlims[0]
        self.y = np.sin(self.x) + np.random.randn(self.x.shape[0])*.2    
    
    def update(self,k=1,n=50):
        plt.scatter(self.x[0:n],self.y[0:n],alpha=1/np.log10(n),zorder =2, label='Data')
        xtemp = np.linspace(self.xlims_ext[0],self.xlims_ext[1],300)
        ytemp = np.zeros_like(xtemp)
        for i in range(xtemp.shape[0]):
            d = (xtemp[i] - self.x[0:n])**2
            closest = d.argsort()[:k]
            ytemp[i] = self.y[closest].mean()
        plt.plot(xtemp,ytemp,'orange',zorder =3, label='Learned f')
        plt.grid(zorder=0)
        plt.xlim(self.xlims_ext)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'1D K-nearest neightbors regression with n={self.n}, k={k}')
        # plt.title(f'Can you guess a good value for a new data point wiht x=5?')
        plt.legend()
        plt.show()

    def interact(self):
        interact(self.update, 
                 k=widgets.IntSlider(min=1., max=50., step=.1, value=1.),
                 n=widgets.IntSlider(min=10., max=100., step=1, value=50.))


class knn_2d_reg(object):
    def __init__(self):
        super().__init__()
        np.random.seed(0)
        self.n=200
        self.x1 = np.random.randn(self.n)
        self.x2 = np.random.randn(self.n)
        self.y  = self.func(self.x1, self.x2)

        self.xlims1 = [self.x1.min(), self.x1.max()]
        self.xlims1_range = self.xlims1[1] - self.xlims1[0]
        self.xlims1_ext = [self.xlims1[0]-.1*self.xlims1_range, self.xlims1[1]+.1*self.xlims1_range]
        self.xlims2 = [self.x2.min(), self.x2.max()]
        self.xlims2_range = self.xlims2[1] - self.xlims2[0]
        self.xlims2_ext = [self.xlims2[0]-.1*self.xlims2_range, self.xlims2[1]+.1*self.xlims2_range]
    
    def func(self, x1, x2):
        return np.sin(x1/2) + np.sin(x2/2)
    
    def update(self,k=1,n=100):
        plt.figure(figsize=[15,10])
        #Ground truth
        ax = plt.subplot(1,2,2,projection='3d',computed_zorder=False)

        x1temp = np.linspace(self.xlims1_ext[0],self.xlims1_ext[1],20)
        x2temp = np.linspace(self.xlims2_ext[0],self.xlims2_ext[1],20)
        X1, X2 = np.meshgrid(x1temp, x2temp)
        X1 = X1.ravel() # make the array 1D
        X2 = X2.ravel()
        Z = self.func(X1,X2)
        surf = ax.plot_trisurf(X1, X2, Z, cmap=cm.coolwarm,
                               linewidth=0,antialiased=False,alpha=0.5,zorder=5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('y')
        zlims = [Z.min(), Z.max()]
        ax.set_zlim(zlims)
        plt.title(f'Ground truth function')
        clb = plt.gcf().colorbar(surf,pad=.1, shrink=0.5, aspect=10)
        clb.ax.set_title('Ground Truth')


        
        ax = plt.subplot(1,2,1,projection='3d',computed_zorder=False)

        x1temp = np.linspace(self.xlims1_ext[0],self.xlims1_ext[1],20)
        x2temp = np.linspace(self.xlims2_ext[0],self.xlims2_ext[1],20)
        X1, X2 = np.meshgrid(x1temp, x2temp)
        X1 = X1.ravel() # make the array 1D
        X2 = X2.ravel()
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                d = (X1[i] - self.x1[0:n])**2 + (X2[i] - self.x2[0:n])**2
                closest = d.argsort()[:k]
                Z[i] = self.y[closest].mean()
        surf = ax.plot_trisurf(X1, X2, Z, cmap=cm.coolwarm,
                               linewidth=0,antialiased=False,alpha=0.5,zorder=5)


        
        ax.scatter(xs=self.x1[0:n],ys=self.x2[0:n],zs=self.y[0:n],color='blue',label='Data')
        plt.xlim(self.xlims1_ext)
        plt.ylim(self.xlims2_ext)
        ax.set_zlim(zlims)
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('y')
        plt.title(f'2D K-nearest neightbors regression with n={self.n}, k={k}')
        clb = plt.gcf().colorbar(surf,pad=.1, shrink=0.5, aspect=10)
        clb.ax.set_title('Learned f')
        plt.legend()
        
        plt.show()

    def interact(self):
        interact(self.update, 
                 k=widgets.IntSlider(min=1., max=20., step=.1, value=0.),
                 n=widgets.IntSlider(min=10., max=100., step=1, value=50.))


class knn_1d_class(object):
    def __init__(self):
        super().__init__()
        np.random.seed(0)
        self.n = 100
        self.x0 = np.random.randn(self.n//2)-2
        self.x1 = np.random.randn(self.n - self.x0.shape[0])+0.5
        self.x = np.concatenate([self.x0, self.x1])
        self.y = np.concatenate([np.zeros_like(self.x0), np.zeros_like(self.x1)+1])

        ind = np.random.permutation(self.n)
        self.x = self.x[ind]
        self.y = self.y[ind]

    def pred(self,x,y,newx,k):
        ytemp = np.zeros_like(newx)
        for i in range(newx.shape[0]):
            d = (newx[i] - x)**2
            closest = d.argsort()[:k]
            ytemp[i] = self.mode(y[closest])
        return ytemp

    def mode(self,x):
        res = np.unique(np.array(x),return_counts=True)
        ind = res[1].argmax()
        return res[0][ind]
    
    def plot_labels(self,x,y,k):
        acc = 0
        pred = self.pred(x,y,x,k)
        for i in range(x.shape[0]):
            # plt.plot([x[i],x[i]],[y[i],x[i]*m_hat+b_hat],color='r',zorder=1,label=label)
            acc += y[i] == pred[i]

            if y[i] == 0:
                marker = 'o' if pred[i] == 0 else 'x'
                plt.scatter([x[i]],[0],marker=marker,color='blue',zorder =2)
            else:
                marker = 'o' if pred[i] == 1 else 'x'
                plt.scatter([x[i]],[100],marker=marker,color='orange',zorder =2)
        return acc/x.shape[0]*100
    
    def sigmoid(self,x):
        return 100/(1+np.exp(-x))
    
    def update(self,k,n):        
        tempx = np.linspace(self.x.min(), self.x.max(),100)
        plt.plot(tempx,self.pred(self.x[0:n],self.y[0:n],tempx,k)*100,'green')
        plt.grid(zorder=0)
        plt.xlabel('x')
        plt.gca().set_ylabel('Probability that y=1',color='green')
        plt.gca().tick_params(axis='y', labelcolor='green')
        plt.plot([self.x[0],self.x[0]],[0,100],color='black',alpha=0) #Plot transparent line to fix ylims
        
        
        ax2 = plt.gca().twinx()  # Second y axis
        ax2.set_ylabel('Labels for y')
        err = self.plot_labels(self.x[0:n],self.y[0:n],k)
        plt.title(f'K-nearest neighbors: k={k}, accuracy = {round(err,1)}%')
        plt.yticks([0,100], labels=[0,1], weight='bold')
        plt.gca().get_yticklabels()[0].set_color('blue',) 
        plt.gca().get_yticklabels()[1].set_color('orange')


        #Legend
        legend_elements = [Line2D([0], [0], color='orange', marker='o',lw=0, label='y=1; pred=1'),
                           Line2D([0], [0], color='orange', marker='x',lw=0, label='y=1; pred=0'),
                           Line2D([0], [0], color='blue', marker='o',lw=0,   label='y=0; pred=0'),
                           Line2D([0], [0], color='blue', marker='x',lw=0,   label='y=0; pred=1')]
        plt.gca().legend(handles=legend_elements, loc='upper left')
        
        plt.show()

    def interact(self):
        interact(self.update, 
                 k=widgets.IntSlider(min=1., max=20., step=.1, value=1.),
                 n=widgets.IntSlider(min=10., max=100., step=1, value=50.))


class knn_2d_class(object):
    def __init__(self):
        super().__init__()
        np.random.seed(0)
        self.n = 500
        self.x1_0 = np.random.randn(self.n//3)-1
        self.x1_1 = np.random.randn(self.n//3)-3
        self.x1_2 = np.random.randn(self.n - self.n//3 - self.n//3)-1
        self.x2_0 = np.random.randn(self.n//3)+1
        self.x2_1 = np.random.randn(self.n//3)+2
        self.x2_2 = np.random.randn(self.n - self.n//3 - self.n//3)+3
        self.x1 = np.concatenate([self.x1_0, self.x1_1, self.x1_2])
        self.x2 = np.concatenate([self.x2_0, self.x2_1, self.x2_2])
        self.y = np.concatenate([np.zeros_like(self.x1_0), 
                                 np.zeros_like(self.x1_1)+1, 
                                 np.zeros_like(self.x1_2)+2])

        self.xlims1 = [self.x1.min(), self.x1.max()]
        self.xlims1_range = self.xlims1[1] - self.xlims1[0]
        self.xlims1_ext = [self.xlims1[0]-.1*self.xlims1_range, self.xlims1[1]+.1*self.xlims1_range]
        self.xlims2 = [self.x2.min(), self.x2.max()]
        self.xlims2_range = self.xlims2[1] - self.xlims2[0]
        self.xlims2_ext = [self.xlims2[0]-.1*self.xlims2_range, self.xlims2[1]+.1*self.xlims2_range]

        ind = np.random.permutation(self.n)
        self.x1 = self.x1[ind]
        self.x2 = self.x2[ind]
        self.y = self.y[ind]
    
    def pred(self,x1,x2,y,newx1,newx2,k):
        model = KNeighborsClassifier(n_neighbors=np.min([k,x1.shape[0]]))
        model.fit(np.stack([x1,x2]).T, y)
        return model.predict(np.stack([newx1, newx2]).T)
    
    def mode(self,x):
        res = np.unique(np.array(x),return_counts=True)
        ind = res[1].argmax()
        return res[0][ind]
    
    def plot_labels(self,x1,x2,y,k):
        acc = 0
        pred = self.pred(self.x1,self.x2,self.y,self.x1,self.x2,k)
        for i in range(x1.shape[0]):
            # plt.plot([x[i],x[i]],[y[i],x[i]*m_hat+b_hat],color='r',zorder=1,label=label)
            acc += y[i] == pred[i]

            if y[i] == 0:
                marker = 'o' if pred[i] == 0 else 'x'
                plt.scatter([x1[i]],[x2[i]],marker=marker,color='blue',zorder =2)
            elif y[i] == 1:
                marker = 'o' if pred[i] == 1 else 'x'
                plt.scatter([x1[i]],[x2[i]],marker=marker,color='orange',zorder =2)
            else:
                marker = 'o' if pred[i] == 2 else 'x'
                plt.scatter([x1[i]],[x2[i]],marker=marker,color='red',zorder =2)
        return acc/x1.shape[0]*100
    
    def sigmoid(self,x):
        return 100/(1+np.exp(-x))
    
    def update(self,k,n):   
        plt.xlabel('x1')
        plt.ylabel('x2')

        
        err = self.plot_labels(self.x1[0:n],self.x2[0:n],self.y[0:n],k)
        plt.title(f'K-nearest neighbors: k={k}, accuracy = {round(err,1)}%')


        #Legend
        legend_elements = [Line2D([0], [0], color='red', marker='o',lw=0,   label='y=2; pred=0'),
                           Line2D([0], [0], color='red', marker='x',lw=0,   label='y=2; pred wrong'),
                           Line2D([0], [0], color='orange', marker='o',lw=0, label='y=1; pred=1'),
                           Line2D([0], [0], color='orange', marker='x',lw=0, label='y=1; pred wrong'),
                           Line2D([0], [0], color='blue', marker='o',lw=0,   label='y=0; pred=0'),
                           Line2D([0], [0], color='blue', marker='x',lw=0,   label='y=0; pred wrong')]
        plt.gca().legend(handles=legend_elements, loc='center left', bbox_to_anchor=[1,0.5])

        #Decision boundary
        xtemp = np.linspace(self.xlims1[0], self.xlims1[1], 250)
        ytemp = np.linspace(self.xlims2[0], self.xlims2[1], 180)
        xtemp, ytemp = np.meshgrid(xtemp,ytemp)
        xtemp, ytemp = xtemp.ravel(), ytemp.ravel()
        pred = self.pred(self.x1[0:n],self.x2[0:n],self.y[0:n],xtemp,ytemp,k)
        plt.scatter(xtemp[pred==0], ytemp[pred==0], color='blue', s=1, marker='o',alpha=0.2)
        plt.scatter(xtemp[pred==1], ytemp[pred==1], color='orange', s=1, marker='o',alpha=0.2)
        plt.scatter(xtemp[pred==2], ytemp[pred==2], color='red', s=1, marker='o',alpha=0.2)
        plt.xlim(self.xlims1)
        plt.ylim(self.xlims2)
        plt.grid(zorder=-1)
        
        plt.show()

    def interact(self):
        interact(self.update, 
                 k=widgets.IntSlider(min=1., max=20., step=.1, value=1.),
                 n=widgets.IntSlider(min=10., max=100., step=1, value=50.))


class multinomial1(object):
    def __init__(self):
        super().__init__()
        np.random.seed(0)
        self.n = 50
        self.x0 = np.random.randn(self.n//2)-2
        self.x1 = np.random.randn(self.n - self.x0.shape[0])+0.5
        self.x2 = np.random.randn(self.n - self.x0.shape[0])+5
        self.x = np.concatenate([self.x0, self.x1, self.x2])
        self.y = np.concatenate([np.zeros_like(self.x0), 
                                 np.zeros_like(self.x1)+1,
                                 np.zeros_like(self.x1)+2])

        self.xlims_range = self.x.max() - self.x.min()
        self.xlims = [self.x.min() - .1*self.xlims_range,self.x.max() + .1*self.xlims_range]

    def plot_labels(self,x,y,c0_hat,c1_hat,b0_hat,b1_hat):
        acc = 0
        for i in range(x.shape[0]):
            f0 = self.softmax(x[i],0,c0_hat,c1_hat,b0_hat,b1_hat)
            f1 = self.softmax(x[i],1,c0_hat,c1_hat,b0_hat,b1_hat)
            f2 = self.softmax(x[i],2,c0_hat,c1_hat,b0_hat,b1_hat)
            pred = np.argmax([f0,f1,f2])
            
            acc += y[i] == pred

            if y[i] == 0:
                marker = 'o' if pred == 0 else 'x'
                plt.scatter([x[i]],[0],marker=marker,color='blue',zorder =2)
            elif y[i] == 1:
                marker = 'o' if pred == 1 else 'x'
                plt.scatter([x[i]],[50],marker=marker,color='orange',zorder =2)
            else:
                marker = 'o' if pred == 2 else 'x'
                plt.scatter([x[i]],[100],marker=marker,color='red',zorder =2)
        return acc/x.shape[0]*100
    
    def softmax(self,x,k,c0,c1,b0,b1):
        f0 = np.exp(c0*x+b0)
        f1 = np.exp(c1*x+b1)
        f2 = 1
        denom = 1 + f0 + f1
        probs = [f0,f1,f2]
        return 100*probs[k]/denom
    
    def update(self,c0 = 0.0, c1=0.0, b0=0.0, b1=0.0):        
        tempx = np.linspace(self.xlims[0], self.xlims[1],100)
        plt.plot(tempx,self.softmax(tempx,0,c0,c1,b0,b1),'blue',zorder =3)
        plt.plot(tempx,self.softmax(tempx,1,c0,c1,b0,b1),'orange',zorder =3)
        plt.plot(tempx,self.softmax(tempx,2,c0,c1,b0,b1),'red',zorder =3)
        
        if c0 != c1:
            x_val = (b1-b0)/(c0-c1)
            if self.softmax(x_val,0,c0,c1,b0,b1) >= self.softmax(x_val,2,c0,c1,b0,b1):
                plt.plot([x_val,x_val],[0,100],'g',linestyle='dashed')
        if c1 != 0:
            x_val = -b1/c1
            if self.softmax(x_val,1,c0,c1,b0,b1) >= self.softmax(x_val,0,c0,c1,b0,b1):
                plt.plot([x_val,x_val],[0,100],'g',linestyle='dotted')
        if c0 != 0:
            x_val = -b0/c0
            if self.softmax(x_val,0,c0,c1,b0,b1) >= self.softmax(x_val,1,c0,c1,b0,b1):
                plt.plot([x_val,x_val],[0,100],'g',linestyle='dashdot')
        plt.grid(zorder=0)
        plt.xlabel('x')
        plt.gca().set_ylabel('Probability (%)')
        plt.plot([self.x[0],self.x[0]],[0,100],color='black',alpha=0) #Plot transparent line to fix ylims

        ax2 = plt.gca().twinx()  # Second y axis
        ax2.set_ylabel('Labels for y')
        err = self.plot_labels(self.x,self.y,c0,c1,b0,b1)
        plt.title(f'Doing multinomial regression manually: accuracy = {round(err,1)}%')
        plt.yticks([0,50,100], labels=[0,1,2], weight='bold')
        plt.gca().get_yticklabels()[0].set_color('blue',) 
        plt.gca().get_yticklabels()[1].set_color('orange')
        plt.gca().get_yticklabels()[2].set_color('red')


        #Legend
        legend_elements = [Line2D([0], [0], color='red', marker='o',lw=0,   label='y=2; pred=2'),
                           Line2D([0], [0], color='red', marker='x',lw=0,   label='y=2; pred wrong'),
                           Line2D([0], [0], color='red',   label='Probability y=2'),
                           Line2D([0], [0], color='orange', marker='o',lw=0, label='y=1; pred=1'),
                           Line2D([0], [0], color='orange', marker='x',lw=0, label='y=1; pred wrong'),
                           Line2D([0], [0], color='orange',   label='Probability y=1'),
                           Line2D([0], [0], color='blue', marker='o',lw=0,   label='y=0; pred=0'),
                           Line2D([0], [0], color='blue', marker='x',lw=0,   label='y=0; pred wrong'),
                           Line2D([0], [0], color='blue',   label='Probability y=0')]
        plt.gca().legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.xlim(self.xlims)

            
        plt.show()

    def interact(self):
        interact(self.update, 
                 c0=widgets.FloatSlider(min=-50., max=50., step=.1, value=0.),
                 c1=widgets.FloatSlider(min=-50., max=50., step=.1, value=0.),
                 b0=widgets.FloatSlider(min=-50., max=50., step=.1, value=0.),
                 b1=widgets.FloatSlider(min=-50., max=50., step=.1, value=0.))


class multinomial2(object):
    def __init__(self):
        super().__init__()
        np.random.seed(0)
        self.n = 100
        self.n_mesh = 50
        self.x0_1 = np.random.randn(self.n//3)-2
        self.x1_1 = np.random.randn(self.n//3)+0.5
        self.x2_1 = np.random.randn(self.n - self.x0_1.shape[0]-self.x1_1.shape[0])+5
        self.x1 = np.concatenate([self.x0_1, self.x1_1, self.x2_1])

        self.x0_2 = np.random.randn(self.n//3)-2
        self.x1_2 = np.random.randn(self.n//3)+0.5
        self.x2_2 = np.random.randn(self.n - self.x0_2.shape[0]-self.x1_2.shape[0])+5
        self.x2 = np.concatenate([self.x0_2, self.x1_2, self.x2_2])

        
        self.y = np.concatenate([np.zeros_like(self.x0_2),
                                 np.zeros_like(self.x1_2)+1,
                                 np.zeros_like(self.x2_2)+2])

        lim_percent = .1
        self.x1lims = [self.x1.min(), self.x1.max()]
        self.x1lims = [self.x1lims[0] - lim_percent*(self.x1lims[1] - self.x1lims[0]),self.x1lims[1] + lim_percent*(self.x1lims[1] - self.x1lims[0])]
        self.x2lims = [self.x2.min(), self.x2.max()]
        self.x2lims = [self.x2lims[0] - lim_percent*(self.x2lims[1] - self.x2lims[0]),self.x2lims[1] + lim_percent*(self.x2lims[1] - self.x2lims[0])]


    def softmax(self,x1,x2,k,d0,d1,c0,c1,b0,b1):
        f0 = np.exp(d0*x1+c0*x2+b0)
        f1 = np.exp(d1*x1+c1*x2+b1)
        f2 = 1
        denom = 1 + f0 + f1
        probs = [f0,f1,f2]
        return 100*probs[k]/denom

    def plot_2d(self,d0,d1,c0,c1,b0,b1):
        acc = 0
        f0 = self.softmax(self.x1,self.x2,0,d0,d1,c0,c1,b0,b1)
        f1 = self.softmax(self.x1,self.x2,1,d0,d1,c0,c1,b0,b1)
        f2 = self.softmax(self.x1,self.x2,2,d0,d1,c0,c1,b0,b1)
        pred = np.argmax([f0,f1,f2], axis=0)
        
        for i in range(self.n):
            acc += self.y[i] == pred[i] 
            if self.y[i] == 0:
                marker = 'o' if pred[i] == 0 else 'x'
                plt.scatter([self.x1[i]],[self.x2[i]],marker=marker,color='blue',zorder =2)
            elif self.y[i] == 1:
                marker = 'o' if pred[i] == 1 else 'x'
                plt.scatter([self.x1[i]],[self.x2[i]],marker=marker,color='orange',zorder =2)
            else:
                marker = 'o' if pred[i] == 2 else 'x'
                plt.scatter([self.x1[i]],[self.x2[i]],marker=marker,color='red',zorder =2)
        plt.ylabel('x2')
        plt.xlabel('x1')

        #Legend
        legend_elements = [Line2D([0], [0], color='red', marker='o',lw=0,   label='y=2; pred=2'),
                           Line2D([0], [0], color='red', marker='x',lw=0,   label='y=2; pred wrong'),
                           Line2D([0], [0], color='orange', marker='o',lw=0, label='y=1; pred=1'),
                           Line2D([0], [0], color='orange', marker='x',lw=0, label='y=1; pred wrong'),
                           Line2D([0], [0], color='blue', marker='o',lw=0,   label='y=0; pred=0'),
                           Line2D([0], [0], color='blue', marker='x',lw=0,   label='y=0; pred wrong')]
        plt.gca().legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1.1, 0.5))

        #Decision boundary
        xtemp = np.linspace(self.x1lims[0], self.x1lims[1], 250)
        ytemp = np.linspace(self.x2lims[0], self.x2lims[1], 180)
        xtemp, ytemp = np.meshgrid(xtemp,ytemp)
        xtemp, ytemp = xtemp.ravel(), ytemp.ravel()
        f0 = self.softmax(xtemp,ytemp,0,d0,d1,c0,c1,b0,b1)
        f1 = self.softmax(xtemp,ytemp,1,d0,d1,c0,c1,b0,b1)
        f2 = self.softmax(xtemp,ytemp,2,d0,d1,c0,c1,b0,b1)
        pred = np.argmax([f0,f1,f2],axis=0)
        plt.scatter(xtemp[pred==0], ytemp[pred==0], color='blue', s=1, marker='o',alpha=0.2)
        plt.scatter(xtemp[pred==1], ytemp[pred==1], color='orange', s=1, marker='o',alpha=0.2)
        plt.scatter(xtemp[pred==2], ytemp[pred==2], color='red', s=1, marker='o',alpha=0.2)
            
        plt.xlim(self.x1lims)
        plt.ylim(self.x2lims)
        plt.grid(zorder=-1)
        plt.title(f'Doing multinomial regression manually: accuracy = {round(acc,1)}%')
    
    def update(self,d0,d1,c0,c1,b0,b1):
        self.plot_2d(d0,d1,c0,c1,b0,b1)
        plt.show()

    def interact(self):
        interact(self.update, 
                 d0=widgets.FloatSlider(min=-10., max=10., step=.1, value=0.),
                 d1=widgets.FloatSlider(min=-10., max=10., step=.1, value=0.),
                 c0=widgets.FloatSlider(min=-10., max=10., step=.1, value=0.),
                 c1=widgets.FloatSlider(min=-10., max=10., step=.1, value=0.),
                 b0=widgets.FloatSlider(min=-10., max=10., step=.1, value=0.),
                 b1=widgets.FloatSlider(min=-10., max=10., step=.1, value=0.))



