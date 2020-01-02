# Day 1: Pandas and Large Datasets

*Date: 2nd January 2020*

Source File:

https://data.world/dataquest/mlb-game-logs 

The above link points to the game logs of MLB games. It can be downloaded from the link as a csv file

```
import pandas as pd

data = pd.read_csv(filename)

data.head()
```

![data.head()](/images/data_head.PNG)


```
data.shape

(171907, 161)
```

Let's check the total memory usage by ```pd.info``` 
This method provides information regarding the dataframe including index dtype & column dtypes, non-null values & memory usage.

By setting  ```memory_usage= 'deep'``` The real memory usage is calculated and shown in human readable units.

```
gl.info(memory_usage='deep')

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 171907 entries, 0 to 171906
Columns: 161 entries, date to acquisition_info
dtypes: float64(77), int64(6), object(78)
memory usage: 860.5 MB
```

That's a lot of memory usage! Let's see what is taking up the majority of the space

```
for dtype in ['int64', 'float', 'object']:
    print(dtype)
    selected_dtype = gl.select_dtypes(include=[dtype])
    selected_dtype.shape
    mean_usage = selected_dtype.memory_usage(deep=True).mean() / 1024 ** 2
    total_usage = selected_dtype.memory_usage(deep=True).sum() / 1024 ** 2
    print("Average memory usage {} columns: {:03.2f} MB".format(dtype, mean_usage))
    print("Total memory usage {} columns: {:03.2f} MB".format(dtype, total_usage))

int64
Average memory usage int64 columns: 1.12 MB
Total memory usage int64 columns: 7.87 MB
float
Average memory usage float columns: 1.29 MB
Total memory usage float columns: 100.99 MB
object
Average memory usage object columns: 9.51 MB
Total memory usage object columns: 751.64 MB

```

So the majority of the space is taken by the object dtypes

Let's define a function which will print out the memory footprint of a dataframe OR series

```
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

```


There are different ways by which the data can be stored. 

| dtype | Memory Size | Max/ Min |
|-------|-------------|----------|
|int8/uint8| 1 byte  | -128/+128 OR 0/255|
|bool| 1 byte | True/False|
|float16/int16/uint16| 2 bytes | -32768/+32767 OR 0/65535| 
|float32 / int32 / uint32| 4 bytes | -2147483648/2147483647|
|float64 / int64 / uint64| 8 bytes | 


The above values can be found by using the ```np.finfo``` and ```np.iinfo```

```
for iit in ['float16', 'float32', 'float64']:
    print(np.finfo(iit))
```



So by downcasting we can save memory. We start by separating out the ```int64``` variables and downcast it to an unsigned integer variable. We will use the ```mem_usage``` function defined above to calculate the memory used by the variables (before & after the operation)

```
data_int = data.select_dtypes(include = ['int64'])
print(data_int.shape)

converted_int = data_int.apply(pd.to_numeric, downcast= 'unsigned')

print(mem_usage(data_int))
print(mem_usage(converted_int))
```

Memory usage has dropped by >80%! 

```
(171907, 6)
7.87 MB
1.48 MB
```

Next we check the ```float``` variables & downcast it to numeric. This is done by applying ```pd.to_numeric```
Later we check the memory usage  
```
data_float = data.select_dtypes(include=['float'])
converted_float = gl_float.apply(pd.to_numeric, downcast='float')

print(data_float.shape)
print(mem_usage(data_float))
print(mem_usage(converted_float))
```

Memory usage has dropped by 50% 

```
(171907, 77)
100.99 MB
50.49 MB
```













