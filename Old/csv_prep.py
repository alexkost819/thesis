import os
import csv
import numpy as np


class CSV_Prep(object):
    """
    CSV Prep is an object for preparing and formatting CSVs of data
    """
    def __init__(self, data_folder_name):
        self.logger = logging.getLogger(__name__)

        root = os.path.join(os.getcwd(), data_folder_name)
        if os.path.exists(root):
            self.orig_simulation_data = os.path.join(root, "simulation")
            self.orig_experiment_data = os.path.join(root, "experiment")
            self.output_simulation_data = os.path.join(root, "output_simulation")
            self.output_experiment_data = os.path.join(root, "output_experiment")

        self.cars, self.pressures = self.identify_cars_and_pressures()

        # TODO: Do I need this?
        self.add_pressure_row(self.cars, self.pressures)

    def identify_cars_and_pressures(self):
        cars = []
        pressures = []
        for _, _, files in os.walk(self.orig_experiment_data):
            for file in files
                car = file.split("_")[0]
                if car not in cars:
                    cars.append(car)

                pressure = file.split("_")[1]
                if pressure not in pressures:
                    pressures.append(pressure)

        return cars, pressures

    # method to add pressure row to all csvs
    def add_pressure_row(self, car, pressure, num_trials):
        owd = os.getcwd()
        os.chdir('Data/output_car_pressure')
        writer = csv.writer(open("output_" + car + "_" + str(pressure) + ".csv", 'w'))
        os.chdir(owd)
        os.chdir('Data/original')
        for num in range(1, num_trials):
            reader = csv.reader(open(str(car) + "_" + str(pressure) + "_" + str(num) + ".csv", 'rb'))
            headers = reader.next()
            headers.append("Pressure")
            writer.writerow(headers)
            for row in reader:
                row.append(str(pressure))
                writer.writerow(row)
        os.chdir(owd)

    # method to read the CSV files associated to the pressure, combine into one big csv
    def combine_car_csv(self, car, pressure, num_trials):
        owd = os.getcwd()
        os.chdir('Data/output_car_pressure')
        fout = open("output_" + car + "_" + str(pressure) + ".csv", "a")
        # first file:
        os.chdir(owd)
        os.chdir('Data/original')
        for line in open(str(car) + "_" + str(pressure) + "_1.csv"):
            fout.write(line)
        # now the rest:
        for num in range(2, num_trials):
            f = open(str(car) + "_" + str(pressure) + "_" + str(num) + ".csv")
            f.next()  # skip the header
            for line in f:
                fout.write(line)
            f.close()  # not really needed
        os.chdir(owd)
        fout.close()
        print("Success! Combined {0} data".format(car))

    # method to combine pressure CSVs independent of cars
    def combine_pressure_csv(self, pressure):
        owd = os.getcwd()
        os.chdir('Data/output_pressure')
        fout = open("output_" + str(pressure) + ".csv", "a")
        os.chdir(owd)
        i = 0
        for root, dirnames, filenames in os.walk('Data/output_car_pressure'):
            for filename in filenames:
                if str(pressure) in filename:
                    f = open(os.path.join(root, filename))
                    if i is not 0:
                        f.next()
                        i = 1
                    for line in f:
                        fout.write(line)
                    f.close()
        os.chdir(owd)
        fout.close()
        print("Success! Combined {0} psi data".format(str(pressure)))

    # method to convert data to frequency domain
    def create_fft_trial(self, filename):
        owd = os.getcwd()
        os.chdir('Data/output_car_pressure')
        # [time, xdir, ydir, zdir, force, pressure_label] = np.genfromtxt(filename, delimiter=',', skip_header=1, max_rows=2048)
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        print(data)
        input("quit here")
        xdir = data[:, 1]
        # time, xdir, ydir, zdir, force, pressure_label
        print("Xdir shape: {0}".format(xdir.shape))
        xdir_fft = np.fft.fft(xdir)
        print("FFT size: {0}".format(xdir_fft.shape))
        print("Pressure label size: {0}".format(pressure_label.shape))
        os.chdir(owd)
        temp = input("exit here")

    def _load_from_file(self, filename):
        with open(filename, 'w') as file:
            time =

def main():
    """
    Sup Main!
    :return: None
    """
    cp = CSV_Prep()
    # fft of each trial
    for root, dirnames, filenames in os.walk('Data/output_car_pressure'):
        for filename in filenames:
            if filename.endswith(".csv"):
                cp.create_fft_trial()

    # # combine all csvs by car and pressure
    # pressuresCamry = [19, 21, 23, 25, 27, 30, 33, 36, 40]
    # for pressure in pressuresCamry:
    #     add_pressure_row('Camry', pressure, 3)
    #     combine_car_csv('Camry', pressure, 3)
    #
    # pressuresCruiser = [24, 26, 28, 30, 32, 34, 36, 38, 40]
    # for pressure in pressuresCruiser:
    #     add_pressure_row('Cruiser', pressure, 7)
    #     combine_car_csv('Cruiser', pressure, 7)
    #
    # pressuresKia = [20, 24, 26, 28, 30, 32, 34, 37]
    # for pressure in pressuresKia:
    #     add_pressure_row('Kia', pressure, 7)
    #     combine_car_csv('Kia', pressure, 7)
    #
    # pressuresLumina = [20, 25, 30, 35]
    # for pressure in pressuresLumina:
    #     add_pressure_row('Lumina', pressure, 3)
    #     combine_car_csv('Lumina', pressure, 3)
    #
    # # combine all CSVs to just pressure
    # pressuresAll = list(set(pressuresCamry) | set(pressuresCruiser) | set(pressuresKia) | set(pressuresLumina))
    # for pressure in pressuresAll:
    #     combine_pressure_csv(pressure)

                # cars = ["Camry", "Cruiser", "Kia", "Lumina"]
                # pressuresCam = [19, 21, 23, 25, 27, 30, 33, 36, 40]     # 3 trials
                # pressuresKia = [20, 24, 26, 28, 30, 32, 34, 37]         # 7 trials
                # pressuresCruiser = [24, 26, 28, 30, 32, 34, 36, 38, 40] # 7 trials
                # pressuresLumina = [20, 25, 30, 35]                      # 3 trials


if __name__ == '__main__':
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(DEFAULT_FORMAT)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    main()
