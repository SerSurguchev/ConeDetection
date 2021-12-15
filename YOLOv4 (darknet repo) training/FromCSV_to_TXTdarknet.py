import pandas as pd
import glob

def from_csv_to_txt(dir, data_fr_file):
    lens = len(glob.glob(dir + '*.jpg'))
    for i in range(1,lens):
        j = 5
        name_of_file = data_fr_file.iloc[i, 0]
        name_of_file = name_of_file[:-3] + "txt"
        without = name_of_file.split('.')[0]
        file1 = open(dir + name_of_file, "w")
        c_width = data_fr_file.iloc[i, 2]
        c_height = data_fr_file.iloc[i, 3]
        while True:
            mark1 = data_fr_file.iloc[i, j]
            if pd.isna(mark1):
                break
            mark1 = str(mark1)
            mark1 = mark1[1:-1]
            mark = mark1.replace(',','')
            x, y, h, w = mark.split()
            x_norm = round((float(x) + float(w) / 2) / float(c_width), 6)
            y_norm = round((float(y) + float(h) / 2) / float(c_height), 6)
            h_norm = round(float(h) / float(c_height), 6)
            w_norm = round(float(w) / float(c_width), 6)
            # print(x_norm, y_norm, w_norm, h_norm)
            file1.write(f"0 {x_norm} {y_norm} {w_norm} {h_norm}\n")
            j += 1
        file1.close()


# Uncomment these lines
if __name__ == '__main__':
    # data_fr_file = pd.read_csv(cvs file path)
    # dir = r'# path with cone images'
    # from_csv_to_txt(dir=dir, data_fr_file=data_fr_file)
    pass