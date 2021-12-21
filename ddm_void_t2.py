import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.special import erf
from scipy.integrate import odeint


###INITIAL PARAMETERS

#Physical parameters
#units: time in 10^6 years, length in kpc, mass in 10^15 M_{\odot}
mu=1.989e45
lu=3.085678e19
tu=31557600*1e6
#and other constants
omega_matter = 0.3
omega_lambda = 1.0 - omega_matter
H0 = 67.810
#
cs=299792458*(tu/lu)
gcons= 6.6742e-11*((mu*(tu**2))/(lu**3))
Ho=(tu/(lu))*H0
gkr=3.0*(((Ho)**2)/(8.0*np.pi*gcons))
lb=3.0*omega_lambda*(((Ho)**2)/(cs**2))    #cosmological constant



#Programme parameters
figno = 0

print_plots = True
debug = True
save_data=True

#Model parameters

#decay rate
th = (cs/Ho)*2.0
dec = (1.0/th)

#injection velocity
vin = 0.0
vk =  -(vin/299792.458)

###
acc = dec*vk

#initial density ratio of two fluids
den_rat = 0.0001

#Choose void type:
compensated = False

if (compensated):
    #S-type
    #spatial inhomogenity
    delta0 = -6.0e-3
    sigma = 6.5
    #r0 = sigma
    void_size = 50 #size of void
    print(compensated,delta0,sigma,void_size)
else:
    #R-type
    #spatial inhomogenity
    delta0 = -3.3e-3
    sigma = 12
    #r0 = sigma
    void_size = 120.0 #size of void
    print(compensated,delta0,sigma,void_size)
###

###INITIAL CONIDITIONS:

#initial and final redshift
zi = 1090.0
zf = 0.0                        #wl will vary this

if (debug):
    print(zi,zf)


#Define function to get lcdm time from redshift
def time_lcdm(z):

    rhzo = gkr*((1.0+z)**3 )       #omega factor at redshift
    x = np.sqrt(lb/rhzo)
    arsinh = np.log(x + np.sqrt(x*x + 1))
    to = np.sqrt((4.0/(3.0*lb)))*arsinh
    return to
###
#Get redshift given time assuming lcdm
def redshift_lcdm(time):

    rhb = lb/((np.sinh(time*(np.sqrt(75e-2*lb))))**2)
    zz = (rhb/gkr)**(1/3)
    za = zz - 1
    return za
###


ti = time_lcdm(zi)
tf = time_lcdm(zf)

if (debug):
    print(ti,tf)

#initialise spatial stuff
#number of grid (ie r) points
ng = 100    #1000

spacing_tuple = np.linspace(0,void_size,ng,retstep=True)
spacing = np.array(spacing_tuple)

rgrid = spacing[0,]
rstep = spacing[1,]

if (debug):
    print('rstep = ', rstep)



#Initialisation of variables
rho = np.empty((ng,))
eta = np.empty((ng,))
tht = np.empty((ng,))
shr = np.empty((ng,))
wey = np.empty((ng,))
heat = np.empty((ng,))
vol = np.ones((ng,))
#
alph=np.empty((ng,))

#note: we (try to) keep this order thro out

#Function to get initial LTB void profile:
def ltb_profile(r,delta0,sigma,z0):

#note: r = 0 throws an error.

    Om = omega_matter
    amp = delta0
    ho = Ho/cs
    r0 = sigma
    dlr = 0.2*r0
    Omo = gkr*((z0+1.0)**3)

    if(compensated):

        Am = (1.0/6.0)*Omo
        Ak = (10.0/3.0)*Am

        dta0 = np.tanh((r-r0)/(2.0*dlr))
        dta1 = (1.0 - dta0**2)*(1.0/(2.0*dlr))
        dta2 = -dta0*dta1/dlr

        d0el = amp*0.5*(1.0 - dta0)
        d1el = -amp*0.5*dta1
        d2el = -amp*0.5*dta2

        m = Am*(1 + d0el)*r*r*r
        mr = Am*d1el*r*r*r + 3.0*(m/r)
        mrr = Am*d2el*(r**3)+3.0*Am*d1el*r*r+3.0*(mr/r)-3.0*(m/(r*r))

        k = Ak*d0el*r*r
        kr = Ak*d1el*r*r + 2.0*(k/r)
        krr = Ak*d2el*r*r+2.0*Ak*d1el*r+2.0*kr/r-2.0*(k/(r*r))

    else:

    	sgm = 0.6*r0
    	m0 = (1.0/6.0)*Omo*(r**3)
    	m0r = (3.0/6.0)*Omo*(r**2)
    	m2a = -2.0*r*np.exp(-1.0*((r/sgm)**2))
    	m2b = np.sqrt(np.pi)*sgm*erf(r/sgm)
    	m1 = (1.0/8.0)*Omo*amp*sgm*sgm*(m2a+m2b)
    	m= m0+m1
    	mr = 0.5*r*r*(Omo*(1.0+amp*np.exp(-(r/sgm)**2)))
    	k = (10.0/3.0)*(m-m0)*(1.0/r)
    	kr = (10.0/3.0)*(mr-m0r)*(1.0/r)
    	kr = kr - k/r
    ###

    rr = 1.0
    rt = np.sqrt(2.0*(m/r) - k + (lb/3.0)*r*r)
    rtr = ((mr/r) - (m/(r**2))*rr - 0.5*kr + (lb/3.0)*r*rr)/rt

    #- density
#    rho = 2.0*(mr-3.0*m*epr)/(r*r*(rr-r*epr))      #epr = 0 ???
    rho = 2.0*(mr)/(r*r*(rr))
    #- expansion
#    tht =  (rtr + 2.0*rt*(rr/r) - 3.0*rt*epr)/(rr - r*epr)
    tht =  (rtr + 2.0*rt*(rr/r))/(rr)
    #- shear
#    shr = np.sqrt((1.0/3.0)*(((rtr - rt*(rr/r))/(rr - r*epr) )**2) )
    shr = np.sqrt((1.0/3.0)*(((rtr - rt*(rr/r))/(rr) )**2) )
    #- weyl
#    wey = - (m/r**3) + (1.0/6.0)*rho
    wey = - (m/r**3) + (1.0/6.0)*rho

    return [rho,tht,shr,wey]
#end ltb function definition

#Get vector of derivatives... the governing system
def get_v(y, t):

    v = np.empty((nv,ng))
    x = np.reshape(y,(nv,ng))
    [rho, eta, tht, shr, wey, heat, vol] = x

    #the boundary cases: ASSUME PARTIAL DERIVATIVES VANISH AT R = 0 AND R = INF
    j = 0

    alph[j] = 0.0
    #
    v[0,j] = -rho[j]*tht[j] - dec*rho[j]
    #
    v[1,j] = -eta[j]*tht[j] + dec*rho[j] - heat[j]*alph[j] - 2*heat[j]*acc
    #
    v[2,j] = (-1.0/3.0)*(tht[j]**2) - (1/2)*(rho[j]+eta[j]) - (3/2)*(shr[j]**2) +acc*alph[j] + (acc**2) + lb
    #
    v[3,j] = (-2.0/3.0)*shr[j]*tht[j] - (0.5)*(shr[j]**2) - wey[j] - (1/3)*acc*alph[j] + (2/3)*(acc**2)
    #
    v[4,j] = -1.0*tht[j]*wey[j] - (1/2)*(rho[j]+eta[j])*shr[j] + (3/2)*shr[j]*wey[j] + (1/6)*heat[j]*alph[j] -(2/3)*acc*heat[j]
    #
    v[5,j] = -(shr[j] + (4/3)*tht[j])*heat[j] - (rho[j]+eta[j])*acc
    #
    v[6,j] = tht[j]*vol[j]
    #

    j = (ng-1)

    alph[j] = 0.0
    #
    v[0,j] = -rho[j]*tht[j] - dec*rho[j]
    #
    v[1,j] = -eta[j]*tht[j] + dec*rho[j] - heat[j]*alph[j] - 2*heat[j]*acc
    #
    v[2,j] = (-1.0/3.0)*(tht[j]**2) - (1/2)*(rho[j]+eta[j]) - (3/2)*(shr[j]**2) +acc*alph[j] + (acc**2) + lb
    #
    v[3,j] = (-2.0/3.0)*shr[j]*tht[j] - (0.5)*(shr[j]**2) - wey[j] - (1/3)*acc*alph[j] + (2/3)*(acc**2)
    #
    v[4,j] = -1.0*tht[j]*wey[j] - (1/2)*(rho[j]+eta[j])*shr[j] + (3/2)*shr[j]*wey[j] + (1/6)*heat[j]*alph[j] -(2/3)*acc*heat[j]
    #
    v[5,j] = -(shr[j] + (4/3)*tht[j])*heat[j] - (rho[j]+eta[j])*acc
    #
    v[6,j] = tht[j]*vol[j]
    #

    #THE GENERAL CASE
    for j in range(1,(ng-1)):

        alph[j] = (vol[j+1]-vol[j])/(2*rstep*vol[j])
        #
        v[0,j] = -rho[j]*tht[j] - dec*rho[j]
        #
        v[1,j] = -eta[j]*tht[j] + dec*rho[j] - (heat[j+1]-heat[j])/(2*rstep) - heat[j]*alph[j] - 2*heat[j]*acc
        #
        v[2,j] = (-1.0/3.0)*(tht[j]**2) - (1/2)*(rho[j]+eta[j]) - (3/2)*(shr[j]**2) +acc*alph[j] + (acc**2) + lb
        #
        v[3,j] = (-2.0/3.0)*shr[j]*tht[j] - (0.5)*(shr[j]**2) - wey[j] - (1/3)*acc*alph[j] + (2/3)*(acc**2)
        #
        v[4,j] = -1.0*tht[j]*wey[j] - (1/2)*(rho[j]+eta[j])*shr[j] + (3/2)*shr[j]*wey[j] + (1/6)*heat[j]*alph[j] - (1/3)*(heat[j+1]-heat[j])/(2*rstep) -(2/3)*acc*heat[j]
        #
        v[5,j] = -(shr[j] + (4/3)*tht[j])*heat[j] - (rho[j]+eta[j])*acc
        #
        v[6,j] = tht[j]*vol[j]
        #

    w = np.reshape(v,(nv*ng))
    return w

###

#initial conditions

def initial_conditions(ng, rgrid, delta0, sigma, zi):

    for j in range(1,ng):

        #initial conditions
        [rho[j],tht[j],shr[j],wey[j]] = ltb_profile(rgrid[j],delta0,sigma,zi)
        eta[j] = rho[j]*den_rat
        heat[j] = eta[j]*vk
        vol[j] = 1.0


    x_initi = np.array([rho, eta, tht, shr, wey, heat, vol])
    return x_initi
###
###
def scaled_radius(X,ng,rstep):
    rv = np.empty((ng,))
    radius = 0.0
    for j in range(ng):
        scale = X[6,j]**(1/3)
        radius = radius + rstep*scale*1e-3
        rv[j] = radius

    return rv
###
#testt function to make and save figures
def print_plots_fn(x,t):
    global figno

    rho = x[0,:]
    eta = x[1,:]
    tht = x[2,:]
    shr = x[3,]
    wey = x[4,:]
    heat = x[5,:]

    rv = scaled_radius(x,ng,rstep)
    za = redshift_lcdm(t)

    #plot final density
    plt.figure(figno)
    plt.plot(rv,rho,label="rho")
    plt.title('rho at {}'.format(za))  #Add Title
    plt.xlabel('radius')            #Add Axis Labels
    plt.ylabel('density (units?)')
    plt.legend(loc = 'lower right') #Add Legend
    #plt.show() #Show the image
    plt.savefig('test_fig_{}.eps'.format(figno),dpi=150) #Save the Figure - not working????
    figno += 1    #Increment Figure Counter
    print(figno)


    if (save_data):
        np.savetxt('rho_at_{}.csv'.format(za),[rho],delimiter=',')
        #etc
###

x_initi =  initial_conditions(ng, rgrid, delta0, sigma, zi)
x = x_initi
nv = x.shape[0]             #number of variables

print_plots_fn(x_initi,ti)

if (debug):
    print("X ndim", x.ndim)
    print("X shape", x.shape)
    print("X size:",x.size)
###

y = np.reshape(x,(ng*nv))

if (debug):
    print("y ndim", y.ndim)
    print("y shape", y.shape)
    print("y size:", y.size)
###


x_in = y


##time steps
nt = 1000    # 5000000
tnum = 5

tspan = tf - ti
trange = tspan/tnum

for k in range(1,tnum):
    if(debug):
        print(tnum)
        
    t0 = ti
    t1 = ti + tnum*trange
    tgrid = np.linspace(t0,t1,nt)
    ti = t1

    #
    x_in = y

    #integrate
    sol = odeint(get_v, x_in, tgrid)
###
    y = sol[(nt-1),:]
    x = np.reshape(y,(nv,ng))

    if (print_plots):
        print_plots_fn(x,t1)
    ###
###
