# EVALUATE SPECTRUM ONLY AT THE POINTS NECESSARY

import numpy as np
import matplotlib.pyplot as plt #can be removed later
from scipy.integrate import quad
from scipy import interpolate
import random
import datetime
from scipy.optimize import minimize_scalar
import argparse
import os

# t in s
# m in eV
# dist in Mpc
# E in MeV


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


# DRAW RANDOM EVENTS FROM REGULAR SPECTRUM, SUBTRACT TIME OF FIRST EVENT AND SMEAR THEM
def drawEvents(mass, dist, events):
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

# take into consideration that the absolute arrival times are not known

def ProbFirstHitDist (mass, dist, events):
    # calculate the probability to get the first hit after a certain amount of time 
    # arrival time distribution of all the hits (for a certain mass) - project the E,t spectrum on the t axis

    # calculate first entry - approximate entry at time 0
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

# energy part

def fillTriggerEff():
    #Read in trigger efficiency of the detector
    E, triggEff = np.loadtxt("trigger_efficiency_100keV_steps.txt", usecols=(0, 1), unpack=True)
    f = interpolate.interp1d(E, triggEff)
    return f


def getEnergySpec(mass, dist,events,t,hitDist):
    # interpolate the energy spectrum at the time of interest t
    triggEff = fillTriggerEff()
    energySpectrum = []
    energy = np.arange(0.1,60.0,2.0)
    for e in energy:
        time = getTimeDelay(t, e, mass, dist)
        pUnsmeared = LL_energy_spectrum(e)*convolveHitDistWithLLTimeSpec(time, mass, dist, events,hitDist)*triggEff(e);
        energySpectrum.append(pUnsmeared)
    f = interpolate.interp1d(energy, energySpectrum)
    return f


def convFunctionEnergy(Evar, E, mass, dist, events, energytest):
    # function for the energy convolution
    if (E-Evar) >0.1 and (E-Evar)< 58.0:
        return gauss(Evar, E)*energytest(E-Evar)
    return 0.0


def applyEnergyRes(E, mass, dist, events,energytest):
    # evaluate the convolution at the energy of interest (smear the energy spectrum by convolution with a gaussian)
    I, error = quad(convFunctionEnergy, -30.0, 30.0, args=(E, mass, dist, events,energytest))
    return I

# CALCULATE PROBABILITY
def calcProbability(mass, dist,events,t,E,hitDist):
    # calculate the probability for an event beeing detected at a certain t+E
    energytest = getEnergySpec(mass,dist,events,t,hitDist)
    val = applyEnergyRes(E, mass, dist, events,energytest)
    return val

# LLH
def llh(mass, eventTimes, eventEnergies,dist,events):
    # calculate the likelihood for the events belonging to a certain mass spectrum 
    # use the - log LLH - better for numberical evaluation and minimization
    hitDist = ProbFirstHitDist(mass, dist, events)
    llh = 0.0
    for i in range( len(eventTimes) ):
        llh += np.log( calcProbability(mass,dist,events,eventTimes[i],eventEnergies[i],hitDist) )

    llh*=-1;
    return llh;


# TODO: add noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		description="Simulate SN Spectrum and fit pseudo-experiments.")
    parser.add_argument("-M", "--mass", default=1.0, type=float,
			help="Neutrino mass (in eV)")
    parser.add_argument("-D", "--distance", default=5.0, type=float,
			help="SN Distance (in Mpc)")
    parser.add_argument("-N", "--nevents", default=10.0, type=float,
			help="Number of expected events from the SN")
    parser.add_argument("--nfits", default=1, type=int,
			help="No. of pseudo-experiments to generate and fit.")
    args = parser.parse_args()
   
    mass = args.mass; distance = args.distance; events = args.nevents; nfits = args.nfits
    
    # creat folder to stor results
    if not os.path.exists('DATA_TEST'):
        os.makedirs('DATA_TEST')

    random.seed(datetime.datetime.now()) # TODO: store to be able to reporduce
    print datetime.datetime.now()
    for i in range(nfits):
        # generate pseudo events
        eventTimes, eventEnergies = drawEvents(mass,distance,events)

        # minimize the likelihood
        x_min = minimize_scalar(llh, args=(eventTimes, eventEnergies,distance,events), bounds=(0.0,5.0), method='bounded', options={'disp':1,'xatol':0.005})
        print i, x_min.nfev, x_min.x
        with open("DATA_TEST/masses_"+str(distance)+"Mpc_"+str(events)+"Events_"+str(mass)+"eV_test.txt", "a") as myfile:
            myfile.write(str(float(x_min.x)) + '\n')

    print datetime.datetime.now()
    print 'DONE'

