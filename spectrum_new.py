# EVALUATE SPECTRUM ONLY AT THE POINTS NECESSARY

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import interpolate
import random
import datetime




# gaussian (around 0): error for the energy smearing from photon counts: N is proportional to E -> N = 1/alpha*E -> N=alpha*E -> factor of alpha for sigma2 (sigma2=E)
def gauss(i, sigma2):
    alpha = 2.22
    return 1/np.sqrt(2.0*np.pi*sigma2*alpha)*np.exp(-(i*i)/(2.0*sigma2*alpha))


# TIME
def LL_time_spectrum(t):
    # arrival spectrum of the neutrinos according to the LL-Model - approximated by log-normal-dist
    my = -1.0324
    sigma = 0.9134
    if t <= 0.0:
        return 0.0
    return np.exp( - ( np.log(t)-my)*(np.log(t)-my)/(2*sigma*sigma) ) / (t*sigma*np.sqrt(2*np.pi))

def getDeltaT(E, mass, dist):
    # time shift due to neutrino mass - factor of 51.4 to get the proper units
    tDelta = dist*51.4635*(float(mass)/float(E))*(float(mass)/float(E))
    return tDelta


def getTimeDelay(t, E, mass, dist):
    return t - getDeltaT(E, mass, dist)

def LL_time_spectrum_shifted(t, E, mass, dist):
    # retrun the value of the LL time spectrum at the shifted time
    time = getTimeDelay(t, E, mass, dist)
    if (time <= 0):
        # spectrum not defined below 0 - there are no events before time 0
        return 0.0
    return LL_time_spectrum(time)

# + ENERGY
def LL_energy_spectrum(E):
    # normalized LL energy spectrum - 15.4: average energy, 3.8: betha, 4802: normalization factor
    return np.power(E, 3.8)*np.exp(-(1.0+3.8)*E/15.4)/4802.516160

def LLSpectrumTotal(E, t, mass, dist):
    # 2D arrival time/energy probability for a certain mass/distance - normalized
    return LL_time_spectrum_shifted(t, E, mass, dist)*LL_energy_spectrum(E)


# DRAW RANDOM EVENTS FROM REGULAR SPECTRUM AND SMEAR THEM
def drawEvents(mass, dist, events):
    # random.seed(datetime.now())
    times = []
    energies = []
    number_of_events = int(events)
    events_generated = 0
    while events_generated < number_of_events:
	    time_r = random.uniform(0.0,10.0)
	    energy_r = random.uniform(0.0,60.0)
	    rand = random.uniform(0.0,0.3)
	    if LLSpectrumTotal(energy_r, time_r, mass, dist) > rand:
		    events_generated += 1
		    times.append(time_r)
		    energies.append(energy_r)
    mintime = min(times)
    # subtract the time of the earliest events from all events
    times[:] = [i - mintime for i in times]
    # smear the energy with a gaussian
    energies[:] = [random.gauss(i, np.sqrt(2.22*i)) for i in energies]
    return times, energies


# up to this point regular absolut arrival time specturm
# PLOT - TEST
'''
testt, teste = drawEvents(0.5,1.0,500)
print testt, teste
myArray = [[LLSpectrumTotal(e,t,0.5,1.0) for t in np.arange(0.0, 10., 0.1)] for e in np.arange(0.1, 60., 0.1)]

X = np.arange(0.0, 10., 0.1)
Y = np.arange(0.1, 60., 0.1)
X, Y = np.meshgrid(X, Y)
Z = myArray

# Surface Plot of the arrival distribution
fig = plt.figure()
ax = fig.add_subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(top=0.87)
surf = ax.contourf(X,Y,Z, 8, cmap=plt.cm.jet)
ax.set_xlabel('time [s]', fontsize=19)
ax.set_ylabel('energy [MeV]', fontsize=19)
plt.plot(testt, teste, 'ro')
#ax.set_title('m = '+str(M)+' eV - ' + str(events) + ' events - D = '+str(D)+ ' Mpc \n'+str(det_type), fontsize=19)
ax.xaxis.set_tick_params(labelsize=19, width=2)
ax.yaxis.set_tick_params(labelsize=19, width=2)
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
# defining custom minor tick locations:
ax.xaxis.set_minor_locator(plt.FixedLocator([50,500,2000]))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both',reset=False,which='both',length=8,width=2)
cbar = fig.colorbar(surf, shrink=1, aspect=20, fraction=.12,pad=.02)
cbar.set_label('relative # of events',size=19)
# access to cbar tick labels:
cbar.ax.tick_params(labelsize=19)
#plt.xlim(0.0, 10.0)
plt.ylim(1, 39)
plt.show()
'''

# take into consideration that the absolute arrival times are not known

def ProbFirstHitDist (mass, dist, events):
    # calculate the probability to get the first hit after a certain amount of time 
    # arrival time distribution of all the hits (for a certain mass) - project the E,t spectrum on the t axis
    # calculate first entry
    totalArrivalTimeDist = []
    timeArray = [0]
    I, error = quad(LLSpectrumTotal, 0.1, 60.0, args=(0,mass,dist))
    totalArrivalTimeDist.append(I* 0.0001 )

    timeBins = np.logspace(-4, 1.01, num=300) # LOG scale!
    for i in range(len(timeBins)-1): 
        time = (timeBins[i+1]+timeBins[i])/2.0
        timeArray.append(time)
        # Integrate over the energy part for every time slice from 0.1 to 60MeV
        I, error = quad(LLSpectrumTotal, 0.1, 60.0, args=(time,mass,dist))
        totalArrivalTimeDist.append(I* (timeBins[i+1]-timeBins[i]) ) # timeBins[i+1]-timeBins[i] time step for log bins
    # calculate the cumulative of the arrival probability
    cumulative = np.cumsum(totalArrivalTimeDist, dtype=float)

    finalTimeSpec = []
    # calculate the final time spectrum
    for i in range(len(totalArrivalTimeDist)):
        finalTimeSpec.append( totalArrivalTimeDist[i]*float(events)*np.power((1 - cumulative[i]), events-1) )
    # normalize
    sumToNorm = sum(finalTimeSpec)
    for i in range(len(finalTimeSpec)-1):
        finalTimeSpec[i] = finalTimeSpec[i]/(sumToNorm* (timeBins[i+1]-timeBins[i]) )
    f = interpolate.interp1d(timeArray, finalTimeSpec)
    #norm, err = quad(f, 0.001, 10.0)
    return f


def convFunction(tau, t, mass, dist, events, hitDist):
    if tau > 0. and tau < 10. and (tau+t) >0. and (tau+t)< 10.0:
        return hitDist(tau)*LL_time_spectrum(tau+t)
    return 0.0


def convolveHitDistWithLLTimeSpec(t, mass, dist, events, hitDist):
    # perform the convolution of the arrival time probability with the ll-time spectrum for A CERTAIN TIME
    I, error = quad(convFunction, 0.001, 10.0, args=(t, mass, dist, events, hitDist))
    return I

def convolveHitDistWithLLTimeSpecTEST( mass, dist, events, hitDist):
    # perform the convolution of the arrival time probability with the ll-time spectrum for A CERTAIN TIME
    # first enrty
    timeBins1 = -np.logspace(0.5, -4, num=50)
    timeBins2 = np.logspace(-4, 1.01, num=300)
    timeBins =  np.concatenate((timeBins1,timeBins2))
    timeArray = []
    totalArrivalTimeDistFinal = []
    for i in range(len(timeBins)-1): 
        time = (timeBins[i+1]+timeBins[i])/2.0
        timeArray.append(time)
        # perform the convolution for a certain time
        I, error = quad(convFunction, 0.001, 10.0, args=(time, mass, dist, events, hitDist))
        totalArrivalTimeDistFinal.append(I)
    f = interpolate.interp1d(timeArray, totalArrivalTimeDistFinal)
    return f

# TEST TIME PART - PLOT
'''
testTime = []
testTime2 = []
testTime3 = []
hitDist = ProbFirstHitDist(mass, dist, events)
times = np.logspace(-4, 1.01, num=50)
for t in times:
    time = getTimeDelay(t,5.0, 0.01, dist)
    time2 = getTimeDelay(t,5.0, 0.0, dist)
    time3 = getTimeDelay(t,5.0, 0.05, dist)
    testTime.append(convolveHitDistWithLLTimeSpec(time,5.0,dist,events,hitDist))
    testTime2.append(convolveHitDistWithLLTimeSpec(time2,5.0,dist,events,hitDist))
    testTime3.append(convolveHitDistWithLLTimeSpec(time3,5.0,dist,events,hitDist))

plt.semilogx(times, testTime, 'ro')
plt.semilogx(times, testTime2, 'bo')
plt.semilogx(times, testTime3, 'go')
plt.show()
'''


def fillTriggerEff():
    #Read in trigger efficiency.
    #Note that the trigger efficiency file for the chosen resolution needs
    #to be located in the proper directory.*/
    E, triggEff = np.loadtxt("trigger_efficiency_100keV_steps.txt", usecols=(0, 1), unpack=True)
    f = interpolate.interp1d(E, triggEff)
    return f


def getEnergySpec(mass, dist,events,t,timeConv):
    #hitDist = ProbFirstHitDist(mass, dist, events)
    triggEff = fillTriggerEff()
    energySpectrum = []
    energy = np.arange(5.0,60.0,1.0)
    #timeConv = convolveHitDistWithLLTimeSpecTEST(mass, dist, events,hitDist)
    # calculate the energy spectrum for a certain time
    for e in energy:
        time = getTimeDelay(t, e, mass, dist)
        print 'time delay', time, t,e,mass,dist
        pUnsmeared = LL_energy_spectrum(e)*triggEff(e)*timeConv(time)
        #pUnsmeared = LL_energy_spectrum(e)*timeConv(time)*triggEff(e);
        #pUnsmeared = LL_energy_spectrum(e)*convolveHitDistWithLLTimeSpec(time, mass, dist, events,hitDist)*triggEff(e);
        energySpectrum.append(pUnsmeared)
    #applyEnergyRes(t, distribution, energySpectrum)
    f = interpolate.interp1d(energy, energySpectrum)
    return f

#energytest = getEnergySpec(mass,dist,events,1.0)
def convFunctionEnergy(Evar, E, mass, dist, events, energytest):
    if (E-Evar) >0.1 and (E-Evar)< 59.0:
        return gauss(Evar, E)*energytest(E-Evar)
    return 0.0


def applyEnergyRes(E, mass, dist, events,energytest):
    # evaluate the convolution at the energy of interest
    #smear the energy spectrum by convolution with a gaussian
    I, error = quad(convFunctionEnergy, -30.0, 30.0, args=(E, mass, dist, events,energytest))
    return I

def calcProbability(mass, dist,events,t,E,timeConv):
    # calculate the probability for an event beeing detected at a certain t+E
    energytest = getEnergySpec(mass,dist,events,t,timeConv)
    val = applyEnergyRes(E, mass, dist, events,energytest)
    return val


def llh(eventTimes, eventEnergies,mass,dist,events):
    hitDist = ProbFirstHitDist(mass, dist, events)
    timeConv = convolveHitDistWithLLTimeSpecTEST(mass, dist, events,hitDist)
    llh = 0.0
    for i in range( len(eventTimes) ):
        llh += np.log( calcProbability(mass,dist,events,eventTimes[i],eventEnergies[i],timeConv) )

    llh*=-1;
    return llh;


print datetime.datetime.now()

masst = 1.0
distt = 1.0
eventst = 160.0

random.seed(12)
for i in range(1):
    eventTimes, eventEnergies = drawEvents(masst,distt,10)
    print eventTimes, eventEnergies
    
    print 'llh', llh(eventTimes,eventEnergies,1.0,distt,eventst)
    print 'llh', llh(eventTimes,eventEnergies,1.01,distt,eventst)
    print 'llh', llh(eventTimes,eventEnergies,1.1,distt,eventst)
print datetime.datetime.now()
# Plot-Spectrum - takes a while
'''
time = np.logspace(-2, 1.01, num=30)
energy = np.arange(0.1,40.0,2.0)
#myArrayF = [[calcProbability(mass, dist, events, t, e) for t in time] for e in energy]

X, Y = np.meshgrid(time, energy)
Z = myArrayF

# Surface Plot of the arrival distribution
fig = plt.figure()
ax = fig.add_subplot(111)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(top=0.87)
surf = ax.contourf(X,Y,Z, 8, cmap=plt.cm.jet)
ax.set_xlabel('time [s]', fontsize=19)
ax.set_ylabel('energy [MeV]', fontsize=19)
#ax.set_title('m = '+str(M)+' eV - ' + str(events) + ' events - D = '+str(D)+ ' Mpc \n'+str(det_type), fontsize=19)
ax.xaxis.set_tick_params(labelsize=19, width=2)
ax.yaxis.set_tick_params(labelsize=19, width=2)
ax.xaxis.set_minor_formatter(plt.FormatStrFormatter('%d'))
# defining custom minor tick locations:
ax.xaxis.set_minor_locator(plt.FixedLocator([50,500,2000]))
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both',reset=False,which='both',length=8,width=2)
cbar = fig.colorbar(surf, shrink=1, aspect=20, fraction=.12,pad=.02)
cbar.set_label('relative # of events',size=19)
# access to cbar tick labels:
cbar.ax.tick_params(labelsize=19)
#plt.xlim(0.0, 10.0)
plt.ylim(1, 39)
plt.show()
'''
print 'DONE'
