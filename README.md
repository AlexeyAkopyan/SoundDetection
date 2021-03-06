# Детектирование аудио-шаблона

Данный репозиторий содержит приложение для обнаружения аудио-шаблона в аудио-потоке.

## Краткое описание метода

Обнаружение участков аудио-потока схожего по звучанию с шаблоном производится за счет сравнения аудио-шаблона с участком схожего размера аудио-потока. Из аудио-шаблона сначала вырезаются участки тишины, чтобы оставить только основной звук. Затем шаблон и аудио-поток переводятся в признаки с помощью функции [librosa.feature.mfcc](https://librosa.org/doc/main/generated/librosa.feature.mfcc.html). После производится свертка над признаками аудио-потока, где в качестве ядра выступает матрица признаков шаблона. Каждое значение полученного вектора интерпретируется как мера схожести шаблона и соответствующего участка аудио-потока. 

Более детальное описание метода представлено в [отчете](https://github.com/AlexeyAkopyan/SoundDetection/blob/main/отчет.pdf).

## Содержание 

1. Файл [find_patterns](https://github.com/AlexeyAkopyan/SoundDetection/blob/main/find_patterns.py) содержит основные функции используемые для детектирования аудио-шаблона.
2. Файл [gui](https://github.com/AlexeyAkopyan/SoundDetection/blob/main/gui.py) используется для графического представления приложения. 
3. Файл [run](https://github.com/AlexeyAkopyan/SoundDetection/blob/main/run.py) позволяет запустить программу из консоли.
4. Папка [sounds](https://github.com/AlexeyAkopyan/SoundDetection/tree/main/sounds) содержит примеры аудио-записей. 


## Использование

Доступно 2 способа использования программы. 
1. Консольный ввод
```
run.py [-h] [-p PATTERN] [-a AUDIO] [--sr SR] [--n-mfcc N_MFCC] [-t THRESHOLD] [-q CUT_QUANTILE]
```
Аргументы:

sr - частота дискретизации используемая при записи аудио (sample rate);

n-mfcc - количество мел-кепстральных коэффициентов при преобразовании аудио-записи в признаки;

t - минимальное допустимое значение меры симметричености, чтобы считать что два звука схожи;

q - параметр, используемые при урезании аудио-шаблона.

2. Графическое приложение 
```
python gui.py
```

Доступно задание тех же параметров, что и в консольном вводе. Присутствует возможность как выбрать уже существующую аудио-запись, так и записать в приложении. Запись начнется после нажатия на кнопку Record/Stop и будет продолжаться до тех пока эта же кнопка не будет нажата еще раз, либо пока не истечет максимальное допустимое время (для аудио-шаблона - 1 сек, для аудио-потока - 10 сек). Также в приложении можно прослушать выбранную или только что записанное аудио. 

Некорректно введенные параметры игнорируются. 
