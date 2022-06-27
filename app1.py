import altair as alt
import streamlit as st

import numpy as np
import pandas as pd

import mpl_toolkits


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.integrate import odeint
from scipy.integrate import ode

from PIL import Image


smin=0 
smax=1
ds = 0.01
Ns=int((smax-smin)/ds)
si=np.arange(smin,smax,ds)


######################################
# Parametres relatif au temps de la simulation
#######################################



###################################################################################
####################################################################################
class Hawk:
    #gamma=2
    def __init__(self,Rgamma=0.5,Rmax=0.5,Qmax=1,Rrho=.4,comp_max=1.25,comp_gam=0.5,comp_sigma=6):
        self.Rgamma=Rgamma
        self.Rmax=1
        self.Rrho=Rrho
        self.comp_max=1
        self.comp_gam=1
        self.comp_sigma=comp_sigma
        
         # Description de la prob de gangne: q_gam=0 et qmax=0, prob const =0.5
        self.Qmax=1 # Qmax >=0,  q_gam=0 
        self.q_gam=1 # q in [0,1], q=0 equi-proba, q=1, si s1 est max s2=min, s1 est sur de gagner (q=1)
        
        self.compType='cst' # type de fonction de competition, cst, sym ou asym
        self.QType='cst' # prend trois valeurs 'cst', 'incr' et 'decr'
    def R(self,s):
#Fonction d'allocation d'energie dans la reproduction
        s=np.array(s)
        smin=min(s);
        smax=max(s);
        s_ren=max(s)-min(s);
        
        s=(s-smin)/s_ren
        y=np.meshgrid(self.Rmax*np.ones(len(s)),self.Rmax*np.ones(len(s)))[0]
        #y=np.meshgrid(self.Rmax*np.power(s,self.Rgamma),self.Rmax*np.power(s,self.Rgamma))[0]
        return y
    #np.meshgrid(self.Rmax*np.power(s,self.Rgamma),self.Rmax*np.power(s,self.Rgamma))[0] #
        # Version slimane
        #y=param_R.max*f((s-smin)/s_ren)+param_R.beta;
    
    def C_funcSym(self,x):
        z=self.comp_max*(np.exp(-self.comp_gam*np.abs((x))))
        return(z)
    def C_funcAsym(self,x):
        z=self.comp_max*(np.exp(-self.comp_gam*(x)))
        return(z)
    
    def C(self,s):
## Fonction cout de la competition
#y: tableau size(S1) ligne et  size(S2) col
#S1: vecteur
#S2: vesteur
        s=np.array(s)
     
        s1,s2=np.meshgrid(s,s);
#C=Ceps/18;
#y=C*triu(x-y); #tril(x-y)
        # Cout constant egale a comp_max
        if self.compType=='cst':
            z=np.meshgrid(self.comp_max*np.ones(len(s)),self.comp_max*np.ones(len(s)))[0]
        
        # cout symetrique
        if self.compType=='sym':
            z=self.comp_max*((np.exp(-self.comp_gam*np.abs((s1-s2)/(max(s)-min(s))))-np.exp(-self.comp_gam))/(1-np.exp(-self.comp_gam))) # non normalise
            #z=self.comp_max*(self.R(s)*np.exp(-self.comp_gam*np.abs((s1-s2)/(max(s)-min(s))))) # renormalise /R
        
        # cout Asymetrique
        if self.compType=='asym':
            z=self.comp_max*(((np.exp(-self.comp_gam*((s1-s2)/(max(s)-min(s)))))-np.exp(-self.comp_gam))/(np.exp(self.comp_gam)-np.exp(-self.comp_gam))) # non normalise
            z=np.transpose(z)    
        return z
    

    def Q_func(self,x):
        z=(0.5+self.Qmax*((np.abs(x))**self.q_gam))
        return(z)
    
    def Q(self,s):
#Probabilite pour un individu s de gagner un individu s' 
        s=np.array(s)
        smin=min(s);
        smax=max(s);
        s_ren=max(s)-min(s);
        
        s=(s-smin)/s_ren
        
        
        s1,s2=np.meshgrid(s,s);
        p=((np.abs(s1-s2)/(max(s)-min(s)))**self.q_gam)#0.5+
       # p=self.comp_max*(np.exp(self.comp_gam*((s1-s2)/(max(s)-min(s))))) # non normalise
        if self.QType=='cst':
            y=0.5*np.meshgrid(self.Qmax*np.ones(len(s)),self.Qmax*np.ones(len(s)))[0]# cst
        else: # cas increasing et decreasing
            y=0.5*self.Qmax*((p*(s2<=s1)+(-p)*(s2>s1))+1) # size dependent
            if self.QType=='incr': # cas decr est sans la transpose
                y=np.transpose(y)
        return y
    
    def nom(self):
        y=self.compType+'Q='+str(self.QType)+'_Rmax='+str(self.Rmax)+'_Cmax='+str(self.comp_max)+'_Cgamma='+str(self.comp_gam)+'_Qmax='+str(self.Qmax)+'_Qgamma='+str(self.q_gam)   
        return(y)
## Fonction de répartition de la loi normale 
# ne pas touche mu
#mu=0; 
#y=normpdf(tan(2*x*pi/2),mu,sigma)/max(normpdf(tan(2*[-1:0.1:1]*pi/2),mu,sigma));

##Fonction xlog(x)
#for j=1:size(x,1)
    #for i=1:size(x,2)       
        #if (x(j,i)>0) && (x(j,i)<1) 
        #y(j,i)=-x(j,i)*log(x(j,i));
        #else 
        #    y(j,i)=0;
        #end;
    #end;
#end

def func3(t,y,param):#S,beta,T_l,T_n): #Matrice de chaque stade
    #The Looser pay cost
   
    si=param[0]     
    d=param[1]
    r=param[2]
    sizeDist=param[3]
    
    C=(1-r)*sizeDist +r*np.diag(np.ones(len(si)))
  
    #A-B/2
    AA=C*(d.R(si)*(d.Q(si)-0.5)-(1-d.Q(si))*d.C(si))
    #B
    B=C*d.R(si)
    
 
    return(y*(np.ones(len(y))-y)*(AA.dot(y)+0.5*B.dot(np.ones(len(y)))))


###################################################################################
####################################################################################

# Fixer Rmax=0.5 dans le cas ou R est constant 
# Fixer Rmax=1 dans le ou R est s dependant

# s=si
def sizeDistribution1(si):
    return(np.exp(-si))

def sizeDistribution2(si):
    a=0.1
    b=1.9
    z=((b-a)/(0.5)**2)*(si-0.5)**2+a

    return(np.array(z)/np.sum(z))

def sizeDistribution3(si):
    a=0.1+1.75
    b=1.9
    z=-((b-a)/(0.5)**2)*(si-0.5)**2+a

    return(z/np.sum(z))

def sizeDistribution4(si):
    z=si
    return(z)
def sizeDistribution5(si):
    z=-si+np.max(si)
    return(z/np.sum(z))
def sizeDistribution6(si):
    b=1
    z=1/len(si)*np.ones(len(si))
    return(z/np.sum(z))
################################


NBAnneeSim=1
tmax=500
dt=.01 # precision de calcul sur le temps (en jour)

timeSet=np.arange(0,tmax,dt)
tmax=10000

hawk=Hawk()

def Hk(s,k=20):
    return  0.8 
y0=[float(Hk(s=s)) for s in si]



################################"
## Interface
## Basic setup and app layout

# Set the configs
APP_TITLE = "Dove-Hawk size dependend"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=Image.open("./utils/logo_bims.png"),
    layout="centered",
    initial_sidebar_state="auto",
)
icon = Image.open("./utils/logo_bims.png")

#st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called


column1, column2,  column4, column5 = st.columns(4)
with column1:
    r_max_input = st.number_input(
        "Resoource max",
        min_value=0.0,
        max_value=None,
        value=0.4,
        step=0.01,
        help="This case is to introduce yor ** R_max ** parameter's value",
    )

    r_input = st.number_input(
        "distance to random",
        min_value=0.0,
        max_value=1.,
        value=0.,
        step=0.01,
        help="This case is to introduce yor ** r ** parameter's value",
    )


with column2:
    st.subheader("Probability to win:")
    q_input = st.radio(
     "Select the probability  Type that you will use ",
     ('constant', 'incresing', 'decresing'))
    
    if q_input == 'constant':
        q_type='cst'
    elif q_input == 'incresing':
        q_type='incr'
    else :
        q_type='decr'

    q_gam_input = st.number_input(
        "Gain probability: gam",
        min_value=0.0,
        max_value=None,
        value=1.0,#64990512
        step=0.01,
        help="This case is to introduce yor ** q_gam ** parameter's value",
    )
    q_max_input = st.number_input(
        "Gain probability: max",
        min_value=0.0,
        max_value=None,
        value=1.0,
        step=0.01,
        help="This case is to introduce yor ** q_max ** parameter's value",
    )
    
    if st.button('See prob.to win'): 
        hawk.q_gam=q_gam_input
        hawk.Qmax=q_max_input
        hawk.QType =q_type
        #Fig 3D
        S1, S2 = np.meshgrid(si,si)
        fig = plt.figure(figsize=(10,10))
                #surface_plot without colorbar FIG 1
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S1, S2, hawk.Q(si), cmap=cm.coolwarm,linewidth=0, antialiased=False)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(10,10)
        st.pyplot(fig)
        
        #######################
        # Fig 2D
        fig1, ax2 = plt.subplots()
        CS = ax2.contourf(S1, S2, hawk.Q(si), 10, cmap=plt.cm.bone)
        
        CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r')   
        cbar = fig1.colorbar(CS)
        cbar.ax.set_ylabel('Prob. to win')
        # Add the contour line levels to the colorbar
        cbar.add_lines(CS2)
        
        st.pyplot(fig1)




    
with column4: 
    st.subheader("Competition cost when loosing:")
    comp_max_input = st.number_input(
        "competition max",
        min_value=0.0,
        max_value=None,
        value=1.0,#64990512
        step=0.01,
        help="This case is to introduce yor ** comp_max ** parameter's value",
    )
    comp_gam_input = st.number_input(
        "competition gam",
        min_value=0.0,
        max_value=None,
        value=0.1,
        step=0.01,
        help="This case is to introduce yor ** comp_gam ** parameter's value",
    )

    compType_input = st.radio(
     "Select the competition Type that you will use ",
     ('constant', 'symmetric', 'asymmetric'))
    if compType_input == 'constant':
        compType='cst'
    elif compType_input == 'symmetric':
        compType='sym'
    else :
        compType='asym'
    
    if st.button('See competition cost'): 
        hawk.compType=compType
        hawk.comp_max=comp_max_input
        hawk.comp_gam=comp_gam_input
        S1, S2 = np.meshgrid(si,si)
        fig = plt.figure()
        #surface_plot without colorbar FIG 1
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S1, S2, hawk.C(si),    cmap=cm.coolwarm,linewidth=0, antialiased=False)


        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(10,10)

        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
        st.pyplot(fig)

        fig1, ax2 = plt.subplots()
        CS = ax2.contourf(S1, S2, hawk.C(si), 10, cmap=plt.cm.bone)
        cbar = fig1.colorbar(CS)
        cbar.ax.set_ylabel('competition cost')
        st.pyplot(fig1)
#######################




with column5:
    st.subheader("Size distribution:")
    sizeDistribution_input = st.radio(
     "Select the size distribution that you will use ",
     ('1: exp', '2: )', '3: (', '4: increasing', '5: decreasing', '6: constant'),index=0)
    if sizeDistribution_input=='1: exp':
         sizeDistribution=sizeDistribution1
    if sizeDistribution_input=='2: )':
         sizeDistribution=sizeDistribution2
    if sizeDistribution_input=='3: (':
         sizeDistribution=sizeDistribution3
    if sizeDistribution_input=='4: increasing':
        sizeDistribution=sizeDistribution4
    if sizeDistribution_input=='5: decreasing':
        sizeDistribution=sizeDistribution5
    if sizeDistribution_input=='6: constant':
        sizeDistribution=sizeDistribution6

    if st.button('See size distribution'): 
        fig = plt.figure()
        plt.plot(sizeDistribution(si))
        st.pyplot(fig)
#
def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")
        
space(1)


if st.button('Run The Model'):  


###############################
    hawk=Hawk()
    r= r_input
    hawk.Rmax=r_max_input
    #probability to win parameters
    hawk.QType=q_type
    hawk.q_gam=q_gam_input
    hawk.q_max=q_max_input
    #Competition parameters
    hawk.compType=compType
    hawk.comp_max=comp_max_input
    hawk.comp_gam=comp_gam_input
    #size distribution 
   # sizeDistribution_list =[sizeDistribution1,sizeDistribution2, sizeDistribution3, sizeDistribution4, sizeDistribution5, sizeDistribution6]
    #sizeDistribution=sizeDistribution6
    #sizeDistribution_list[int(sizeDistribution_input[-1])-1]
    #1: exp increasing, 2: ), 3:(, 4:increasing, 5: decreasing, 6: cst
    
    param=(si,hawk,r,sizeDistribution(si)) 


    sol = ode(func3).set_integrator('lsoda',rtol=0.01)
    sol.set_initial_value(y0,0).set_f_params(param) #param: doit etre un tableau
    #sol=odeint(func, y0, temps, args=param) 
    
    #res=sol.integrate(10)
    
    res1=[] # tableau des resultats
    timeSet=[] # tableau des temps 
    while sol.successful() and sol.t < tmax:
        ss=sol.integrate(sol.t+dt)
        if sol.t>tmax-365:
            timeSet.append(sol.t+dt)
            res1.append(ss)
    
    res1=np.transpose(np.array(res1))
    d = {'Dove & Hawak at t = t_max' : pd.Series(res1[:,-1])}
    # creates Dataframe.
    df = pd.DataFrame(d)
    
    #-----------------------------------------------------------------------------#
    #----------------------------         plot          --------------------------#
    #-----------------------------------------------------------------------------#
    
    #st.line_chart(df)
    st.line_chart(df[df.columns[0]])
   
    
   
        
    #%%  %%%%%%%%%%%%%%%%%%%%%%%%%
    #Figures des resultats
    #####################################    
    
    T, S = np.meshgrid(timeSet, si)
    
    fig = plt.figure()
    #surface_plot without colorbar FIG 1
    ax = fig.add_subplot(111, projection='3d')
    
    
    
    surf = ax.plot_surface(S, T, res1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    #plt.title('hawk Population Density')
    plt.xlabel(' s')
    plt.ylabel(' t')
    # normalize 0..1
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(70,10)
    
    st.pyplot(fig)
    # #x.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
    # #            r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
    # ####################
    
    # ###################################
    
    fig1, ax2 = plt.subplots()
    CS = ax2.contourf(S, T, res1)#, 10, cmap=plt.cm.bone)
    
    #CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r')
    
    plt.xlabel('s',fontsize=12)
    plt.ylabel('time',fontsize=12)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig1.colorbar(CS)
    cbar.ax.set_ylabel(r"$H(t,s)$",fontsize=12)
    # Add the contour line levels to the colorbar
    #cbar.add_lines(CS)
    
    #plt.savefig('DH3D'+outName+'.pdf', transparent=True)
    st.pyplot(fig1)
    
    


if st.button('Run r-effect (distance to random)'):  
    hawk=Hawk()
    hawk.Rmax=r_max_input
    #probability to win parameters
    hawk.QType=q_input
    hawk.q_gam=q_gam_input
    hawk.q_max=q_max_input
    #Competition parameters
    hawk.compType=compType
    hawk.comp_max=comp_max_input
    hawk.comp_gam=comp_gam_input
    #size distribution 
    sizeDistribution_list =[sizeDistribution1,sizeDistribution2, sizeDistribution3, sizeDistribution4, sizeDistribution5, sizeDistribution6]
    sizeDistribution=sizeDistribution6
    #sizeDistribution_list[int(sizeDistribution_input[-1])-1]
    #1: exp increasing, 2: ), 3:(, 4:increasing, 5: decreasing, 6: cst
    
    
    
    
    rSet=np.linspace(0,1,10)
    res2=[]
    for rr in rSet:
    
        param=(si,hawk,rr,sizeDistribution(si)) 
        sol = ode(func3).set_integrator('lsoda',rtol=0.01)
        sol.set_initial_value(y0,0).set_f_params(param) #param: doit etre un tableau
    #sol=odeint(func, y0, temps, args=param) 
    
    #res=sol.integrate(10)
    
        res1=[] # tableau des resultats
        timeSet=[] # tableau des temps 
        while sol.successful() and sol.t < tmax:
            ss=sol.integrate(sol.t+dt)
            if sol.t>tmax-365:
                timeSet.append(sol.t+dt)
                res1.append(ss)
    
        res1=np.transpose(np.array(res1))
    
        res2.append(res1[:,-1])
    
     
    res2=np.transpose(res2)
    res2[res2<0]=0
    
    #%%  %%%%%%%%%%%%%%%%%%%%%%%%%
    # Figures des resultats
    ######################################    
    
    T, S = np.meshgrid(rSet, si)
    
    fig = plt.figure()
    #surface_plot without colorbar FIG 1
    ax = fig.add_subplot(111, projection='3d')
    
    
    
    surf = ax.plot_surface(S, T, res2, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    #plt.title('hawk Population Density')
    plt.xlabel(' s')
    plt.ylabel(' r')
    # normalize 0..1
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(70,10)
    st.pyplot(fig)
    
    #x.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
    #            r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')
    ####################
    
    ###################################
    
    fig1, ax2 = plt.subplots()
    CS = ax2.contourf(S, T, res2)#, 10, cmap=plt.cm.bone)
    
    #CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r')
    
    plt.xlabel('s',fontsize=12)
    plt.ylabel('r',fontsize=12)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig1.colorbar(CS)
    cbar.ax.set_ylabel(r"$H(t,s)$",fontsize=12)
    
    # Add the contour line levels to the colorbar
    #cbar.add_lines(CS)
    
    st.pyplot(fig1)

















    
