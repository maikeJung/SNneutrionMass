import spectrum
from scipy.optimize import minimize
from scipy.optimize import brent
from scipy.optimize import minimize_scalar
from scipy.optimize import fmin
from scipy.interpolate import interp1d
from scipy import interpolate
import numpy as np
from datetime import datetime
import argparse
import os
from numpy import transpose
import random
import matplotlib.pyplot as plt

def llh(massi):
    # calculate LLH
    spectrumTest = spectrum.doubleArray( (RESE - 1) * REST )
    spectrum.createSpectrum(spectrumTest, float(massi), distance, events, useEnergyRes, useTriggerEff, noise, noise_events);

    #Interpolate
    timeArray = range(0, REST, 1)
    energyArray = range(0, RESE-1)
    myArray = [[spectrumTest[t*(RESE-1) +e] for t in timeArray] for e in energyArray]
    #tint = np.asarray(np.arange(0, TMAX, STEPT))
    tint = [timeArray[i]*STEPT for i in timeArray]
    eint = [energyArray[i]*STEPE + STEPE for i in energyArray]
    #eint2 = np.asarray(np.arange(STEPE, EMAX-STEPE, STEPE))
    valint = transpose(myArray)
    interpf = interpolate.RectBivariateSpline(tint, eint, valint, kx=3, ky=3, s=0)

    '''
    myArray = [[spectrumTest[t*(RESE-1) +e] for t in range(0, REST, 1)] for e in range(0, RESE-1)]
    tint = np.asarray(np.arange(0, TMAX, STEPT))
    eint = np.asarray(np.arange(STEPE, EMAX-STEPE, STEPE))
    valint = transpose(myArray)
    interpf = interpolate.RectBivariateSpline(tint, eint, valint, kx=3, ky=3, s=0)
    '''

    llh = 0.0
    for i in range( int(events) ):
        if (interpf(eventTime[i],eventEnergy[i]) < pow(10,-200)):
            llh += -10000000  
            printf("event number %d e %d t %d\n",i, eventEnergy[i], eventTime[i])
            printf("value of spectrum very small - check \n")
        else:
            #llh += np.log( interpf(eventEnergy[i],eventTime[i]) )
            llh += np.log( interpf(eventTime[i],eventEnergy[i]) )

    llh*=-1;
    return llh;

def createHist(masses, bin_width):
    # store determined masses in a histogram from 0 to 5 eV
    events = len(masses)
    # store values in histogram -> values_pseudo: values of bins; mass_hist_pseudo: center of the bins
    bins = np.arange(0.0 - bin_width/2.0, 5.0, bin_width)
    values_pseudo, m = np.histogram(masses, bins=bins)
    mass_hist_pseudo = np.arange(0.0, 5.0 - bin_width/2.0, bin_width)
    return mass_hist_pseudo, values_pseudo/float(events)

def calcError(masses):
    # calculate the 1 sigma uncertainty, by checking where 68% of events are detected
    mass_hist, values = createHist(masses, 0.001)
    mass_hist = mass_hist[::-1]
    values = values[::-1]
    values = np.cumsum(values)
    f = interp1d(values, mass_hist)
    return f(0.84135), f(0.5), f(0.15865)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
		description="Simulate SN Spectrum and fit pseudo-experiments.")
    parser.add_argument("-M", "--mass", default=1.0, type=float,
			help="Neutrino mass (in eV)")
    parser.add_argument("-D", "--distance", default=5.0, type=float,
			help="SN Distance (in Mpc)")
    parser.add_argument("-N", "--nevents", default=10.0, type=float,
			help="Number of expected events from the SN")
    parser.add_argument("--perfect-trigger", dest='triggEff',
			default=True, action='store_false',
			help="Assume fully  eff. trigger across all energies.")
    parser.add_argument("--perfect-reco", dest='energyRes', default=True,
			action='store_false', help="Assume perfect energy reco.")
    parser.add_argument("--nfits", default=1, type=int,
			help="No. of pseudo-experiments to generate and fit.")
    parser.add_argument("--noiseb", default=0.01, type=float,
			help="Noise - exponential function")
    args = parser.parse_args()
    
    #TODO also import? also parse noise?
    RESE = 600
    REST = 1000
    EMAX = 60.0
    TMAX = 10.0
    STEPE = EMAX/RESE
    STEPT = TMAX/REST

    noise = pow(10,-3)*STEPT
    noise_events = 0.01

    mass = args.mass; distance = args.distance; events = args.nevents; nfits = args.nfits
    useTriggerEff = args.triggEff; useEnergyRes = args.energyRes
    #noise = args.noiseb

    #draw events and noise events from Poisson distribution
    #neutrinoEvents = np.random.poisson(events)
    #noiseEvents = np.random.poisson(noise_events)
    #events = neutrinoEvents + noiseEvents


    # create spectrum from which the events are drawn
    # TODO: draw number of events from Poisson distribution - probably needs to be created for every event
    spectrumGen = spectrum.doubleArray( (RESE - 1) * REST )
    spectrum.createSpectrum(spectrumGen, mass, distance, events, useEnergyRes, useTriggerEff, noise, noise_events);

    #Interpolate
    timeArray = range(0, REST, 1)
    energyArray = range(0, RESE-1)
    myArray = [[spectrumGen[t*(RESE-1) +e] for t in timeArray] for e in energyArray]
    #tint = np.asarray(np.arange(0, TMAX, STEPT))
    tint = [timeArray[i]*STEPT for i in timeArray]
    eint = [energyArray[i]*STEPE + STEPE for i in energyArray]
    #eint2 = np.asarray(np.arange(STEPE, EMAX-STEPE, STEPE))
    valint = transpose(myArray)
    interpf = interpolate.RectBivariateSpline(tint, eint, valint, kx=3, ky=3, s=0)

    # Plot Test
    '''
    X, Y = np.meshgrid(tint, eint)
    test = []
    for e in eint:
        testi = []
        for t in tint:
            testi.append(float(interpf(t,e)))
        test.append(testi)
    #print test
    
    # Surface Plot of the arrival distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(top=0.87)
    surf = ax.contourf(X,Y,test, 8, cmap=plt.cm.jet)
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
    plt.ylim(1, 59)
    plt.show()
    '''

    # find maximum in the spectrum - needed to draw random events from spectrum
    maxSpectrum = spectrum.findSpectrumMax(spectrumGen)

    if not os.path.exists('DATA'):
        os.makedirs('DATA')
    masses = []
    #random.seed(datetime.now()) # seed random number generatot -> store
    random.seed(3)
    for i in range(nfits):
        # create one pseudo experiment
        eventEnergy = []
        eventTime = []
        events_generated = 0
        while events_generated < events:
	        time_r = random.uniform(0.0,10.0)
	        energy_r = random.uniform(0.0,60.0)
	        rand = random.uniform(0.0,maxSpectrum)
	        if (float(interpf([time_r],[energy_r])) > float(rand) ):
		        events_generated += 1
		        eventTime.append(time_r)
		        eventEnergy.append(energy_r)

        # find the mass for which the likelihood is minimal and store it
        x_min = minimize_scalar(llh, bounds=(0.0,5.0), method='bounded', options={'disp':1,'xatol':0.005})
        print i, x_min.nfev, x_min.x
        masses.append(x_min.x)
        with open("DATA/masses_"+str(distance)+"Mpc_"+str(events)+"Events_"+str(mass)+"eV_"+str(noise_events)+"noiseEvents_"+str(noise)+"Noise_test.txt", "a") as myfile:
            myfile.write(str(float(x_min.x)) + '\n')

    # calculate 1 sigma uncertainty and store
    lower, value, upper = calcError(masses)

    with open("DATA/detection_error_test.txt", "a") as myfile:
        myfile.write(str(distance) +" "+ str(events) +" "+ str(mass) +" " +str(noise)+ " " + str(lower) +" "+ str(value)+" " + str(upper) + '\n')

    print 'DONE'
