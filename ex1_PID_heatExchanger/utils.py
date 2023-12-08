import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import scipy.signal as sgnl
import os

def plotData(t, data, xlabel="time (s)", ylabel="y", titel="", grid=True, labels=None):
    '''
    Visualiseer de data in een figuur.

    Matplotlib's pyplot wordt gebruikt om data te visualiseren in een figuur. Meerdere datasets kunnen weergegeven worden (e.g. plotData([t_1, t_2, ..., t_n], [data_1, data_2, ..., data_n]))

    Parameters:
        t (list): de onafhankelijke variabele
        data (list): de afhankelijke variabele
        xlabel (string): (opt) label van x-as (default: time (s))
        ylabel (string): (opt) label van y-as (default: y)
        titel (string): (opt) figuurtitel (default: "")
        grid (boolean): (opt) voegt een grid toe aan de figuur (default: True)
        labels (list): (opt) per plot kan een label toegevoegd worden voor de legende (default: None)
    '''
    t = _checkData(t)
    data = _checkData(data)
    plt.figure()
    for i in range(len(t)):
        if labels is not None:
            plt.plot(t[i], data[i], label=labels[i])
        else:
            plt.plot(t[i], data[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titel)
    plt.grid(grid)
    if labels is not None:
        plt.legend()
    plt.show()


def _checkData(obj):
    '''
    Controleert of de data objecten correct zijn voor de functie plotData.

    Een data input voor plotData moet een lijst van lijsten zijn. Zo niet, zal het niet mogelijk zijn om de figuur weer te geven. Vooral in de situatie dat er maar één grafiek gemaakt moet worden, worden de lists t en data vaak verkeerdelijk niet nogmaal in een list gestoken. Dat wordt hier gecorrigeerd.

    Parameters:
        obj (list of ints/floats): het object dat nagekeken moet worden.

    Returns:
        obj_correct (list): het gecorrigeerde object.  
    '''
    """
    Algoritme:
    1. Is object een ndarray?
        Ja: transformeer naar list.
            A. Is object geen ndarray van ndarrays?
                Ja: voeg dimensie toe aan list.
        Nee:
    2. Is object een list?
        Ja:
            A. bestaat list uit ndarrays?
                Ja: stapel alle ndarrays op elkaar en maak er een list van.
                Nee:
            B. bestaat list niet uit lists?
                Ja: voeg dimensie toe aan list.
                Nee: Behoud zoals het is.
    """
    if (isinstance(obj, np.ndarray)):
        obj_correct = obj.tolist()
        if np.isscalar(obj_correct[0]):
            obj_correct = [obj_correct]
    elif (isinstance(obj, list)):
        if isinstance(obj[0], np.ndarray): 
            obj_correct = np.stack(tuple(obj), axis=1).T.tolist()
        elif np.isscalar(obj[0]):
            obj_correct = [obj]
        else:
            obj_correct = obj

    return obj_correct

def leesCSV(bestandsNaam, plot=False, seperator=';', decimal=',', header=None, data_kolom=1):
    '''
    Lees CSV bestand.

    Pandas' read_csv functie wordt gebruikt om data uit een CSV file in te lezen.

    Parameters:
        bestandsNaam (string): naam van CSV bestand (voeg .csv extentie toe)
        plot (boolean): (opt) maak een plot van de data (default: False)
        separator (string): (opt) seperator tussen de datapunten (default: ';')
        decimal (string): (opt) decimaal punt (default: ',')
        header (int): (opt) rijnummers van de header, e.g. verwijder rij 0 is 'header=0', None betekent dat er geen header is (default: None)
        data_kolom (int): (opt) de kolom die u wilt importeren. De kolom na de tijdsas is kolom 1 (default: 1)

    Returns:
        t (list): de tijdsvector
        data (list): de data
    '''
    # Create the path to the file - in the parent map
    script_dir = os.path.dirname(__file__)
    file_path_absolute = os.path.join(script_dir, bestandsNaam)
    file_path = os.path.abspath(file_path_absolute)

    # Read CSV file
    data_raw = pd.read_csv(file_path, sep=seperator, decimal=decimal, header=header)

    # Transform to matrix notation
    data_matrix = data_raw.values
    t = data_matrix[:, 0].tolist()
    try:
        data = data_matrix[:, data_kolom].tolist()
    except IndexError:
        error = '[LeesCSV] De index van de data kolom is niet correct. Er zijn maximaal {} datakolommen.'.format(len(data_matrix[0,:]) - 1)
        raise IndexError(error)

    # Plot data
    if plot:
        plotData([t], [data], titel="Data: " + bestandsNaam)
    return t, data


def raaklijn(y, Ts=1.0, plot_verbose=False):
    '''
    Bepaalt de raaklijn in het buigpunt aan een eerste-order stapresponsie.

    Een eerste-orde stapresponsie wordt gefilterd en vervolgens numeriek afgeleid. Het buigpunt wordt bepaald als het punt waar de tweede afgeleide nul wordt. Vervolgens wordt de raaklijn y = q * (t - a) + b getekend.

    Parameters:
        y (list): stapantwoord
        Ts (float): (opt) bemonsteringstijd (default: 1.0)
        plot_verbose (boolean): (opt) visualiseer tussenresultaten (default: False)

    Returns:
        a (float): tijdsverschuiving
        b (float): magnitudeverschuiving
        q (float): richtingscoëfficient
    '''
    # Apply Savitzky-Golay filter to step response
    y_hat = sgnl.savgol_filter(y, 35, 4)
    t = [i*Ts for i in range(len(y))]
    if plot_verbose:
        plotData([t, t], [y, y_hat], titel="Non-filtered vs filtered step response")
    
    # Standard discrete diff
    diff = np.diff(y, append=[y[-1]])
    t_diff = [i*Ts for i in range(len(diff))]

    # Central diff on smoothed signal
    central_diff = np.gradient(y_hat)
    t = [i*Ts for i in range(len(central_diff))]

    if plot_verbose:
        # Plot filtered and unfiltered diff
        plotData([t, t_diff], [central_diff, diff], titel="Filtered central diff vs regular diff")

    # Calculate parameters of tangent line
    max_out = np.max(central_diff) / Ts
    max_index = np.argmax(central_diff)
    a = t[max_index]
    b = y_hat[max_index]

    static_gain = np.mean(y_hat[-10:])
    t_end = (static_gain * 1.10 - b)/max_out + a
    t_raak = [a - 20 * Ts, a, t_end]
    y_raak = [0]*3
    for i in range(3):
        y_raak[i] = b + max_out*(t_raak[i] - a)

    plotData([t, t_raak], [y, y_raak], titel="Step response with tangent line")
    return a, b, max_out