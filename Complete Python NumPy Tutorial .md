```python
pip install numpy

```

    Requirement already satisfied: numpy in c:\users\dell\anaconda3\lib\site-packages (1.20.1)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import numpy as np
```


```python
a=np.array([1,2,3]) #Creating a numpy array
print(a)
#This is one dimensional array.
```

    [1 2 3]
    


```python
b=np.array([[9.0,8.0,7.0],[1.0,2.0,3.0]])
print(b)

#Two dimensional array.
```

    [[9. 8. 7.]
     [1. 2. 3.]]
    


```python
a.ndim #Get dimensions
```




    1




```python
b.ndim
```




    2




```python
a.shape #We get the shape of our numpy array.
```




    (3,)




```python
b.shape
```




    (2, 3)




```python
a.dtype #We are asking what data type is in np array a.
```




    dtype('int32')




```python
#IF we want to create a np array with dtype fixed before

c=np.array([1,2,3,4],dtype='int16') #We are making the size lesser.
c
```




    array([1, 2, 3, 4], dtype=int16)




```python
a.itemsize #Provides us the size per item(bytes)
```




    4




```python
a.size #Returns total number of elements

```




    3




```python
a.nbytes
#Returns the total amount of bytes in the array.
```




    12



# Accessing/Changing specific elements, rows, columns,etc.


```python
d=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(d)
```

    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]]
    


```python
#Get a specified element arr[r,c]

d[1,4] #We get the value 10
d[1,-1] #We get 10.. so the entire indexing thing we can do the way we used to in lists.

```




    10




```python
d[:,2] #So the ":" operator in row kinda indicates that all the rows are to be considered...
```




    array([3, 8])




```python
d[1,:] #So it will print the row at index 1.
```




    array([ 6,  7,  8,  9, 10])




```python
#Getting a little fancy [startindex:endindex:stepsize]
d[0,1:4:2]
d[1,1:-1:2]
```




    array([7, 9])




```python
#Changing/assigning new values for the elements inside np.array

d[1,3]=144
print(d)
```

    [[  1   2   3   4   5]
     [  6   7   8 144  10]]
    


```python
d[:,2]=[1,2] #The dimensions have to be correct in order for this to work
print(d)
```

    [[  1   2   1   4   5]
     [  6   7   2 144  10]]
    

*3-d example


```python
b=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
```

    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    


```python
#Get specific element (work from outside to inside)

b[0,0,1]
```




    2




```python
b[:,1,:]
```




    array([[3, 4],
           [7, 8]])




```python
b[:,1,:]=[[10,9],[11,12]]
print(b)
```

    [[[ 1  2]
      [10  9]]
    
     [[ 5  6]
      [11 12]]]
    

## Initializing Different Types of Arrays


```python
#All 0s matrix

np.zeros(5)#We just need to specify the shape.
```




    array([0., 0., 0., 0., 0.])




```python
np.zeros((2,3,3))#We just need to specify the shape. #We had to do double bracket for this.
```




    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]])




```python
#An array of all ones
np.ones((4,2,2),dtype='int32') #We are able to add seperate paramenters to it.
```




    array([[[1, 1],
            [1, 1]],
    
           [[1, 1],
            [1, 1]],
    
           [[1, 1],
            [1, 1]],
    
           [[1, 1],
            [1, 1]]])




```python

#For any other number
np.full((2,2),101)
np.full(b.shape,14) #Assigned the shape that we wanted.. which was previously made.
```




    array([[14, 14, 14],
           [14, 14, 14]])




```python
#Other number full like method

#We can reuse an already build set

np.full_like(a,101)
```




    array([101, 101, 101])




```python
 #Random decimal numbers

np.random.rand(4,2)#We don't use the outer () it then gives us an error regarding tuple
```




    array([[0.3057879 , 0.16913456],
           [0.02079943, 0.44095939],
           [0.62275631, 0.52540708],
           [0.77723383, 0.48382087]])




```python
np.random.rand(a.shape)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-19-2ef92a0b90f7> in <module>
    ----> 1 np.random.rand(a.shape)
    

    mtrand.pyx in numpy.random.mtrand.RandomState.rand()
    

    mtrand.pyx in numpy.random.mtrand.RandomState.random_sample()
    

    _common.pyx in numpy.random._common.double_fill()
    

    TypeError: 'tuple' object cannot be interpreted as an integer



```python
np.random.random_sample(a.shape) #In here we can mention the shape like we did previously using a tuple.
```




    array([0.64943037, 0.15624492, 0.93154043])




```python
#Random integer values

np.random.randint(-1,9,size=(2,3)) #We have to mention the start and the end range.. so yeah the last value is not considered.. just like python range.. and we have to mention the shape seperately interms of size.
```




    array([[ 2,  3, -1],
           [ 3,  5, -1]])



# The identity matrix


```python
np.identity(5)
```




    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])




```python
#Repeating an array

arr=np.array([1,2,3])
r1=np.repeat(arr,3,axis=0) #The axis here does not really affect the code much as it is a one dimensional array
print(r1)
```

    [1 1 1 2 2 2 3 3 3]
    


```python
arrr=np.array([[1,2,3]])
r2=np.repeat(arrr,3,axis=0) #There is no effect of axis here at value of axis=0
print(r2)
```

    [[1 2 3]
     [1 2 3]
     [1 2 3]]
    


```python
arrr=np.array([[1,2,3]])
r2=np.repeat(arrr,3,axis=1)
print(r2) #The value of axis=1 makes it into an entirely straight line.
```

    [[1 1 1 2 2 2 3 3 3]]
    

# So he told us to make a particular array based on what we have learnt


```python
ans=np.zeros((5,5))
print(ans)

print(" ")

ans[0,:]=1
ans[4,:]=1
ans[:,(0,4)]=1
ans[2,2]=9

print(ans)

#So I got the exact one with the code above.
```

    [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]
     
    [[1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 1.]
     [1. 0. 9. 0. 1.]
     [1. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1.]]
    


```python
#This is his version of solution

output=np.ones((5,5))
print(output)
```

    [[1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1.]]
    


```python
z=np.zeros((3,3))
print(z)

print(" ")
z[1,1]=9
print(z)
```

    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
     
    [[0. 0. 0.]
     [0. 9. 0.]
     [0. 0. 0.]]
    


```python
output[1:4,1:4]=z
print(output)
```

    [[1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 1.]
     [1. 0. 9. 0. 1.]
     [1. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1.]]
    

# Be careful when copying arrays


```python
a=np.array([1,2,3])
b=a.copy()

b[0]=100
print(b)


#Had I not put copy(), then what would've happened is that the b variable would've just pointed to a.. so any 
#changes to b would've altered a.
```

    [100   2   3]
    

# Mathematics


```python
a=np.array([1,2,3])
print(a)

#Element wise +,-,/..
```

    [1 2 3]
    


```python
a+2 #Every element got added by two.
```




    array([3, 4, 5])




```python
a-2
```




    array([-1,  0,  1])




```python
a*2
```




    array([2, 4, 6])




```python
a/2
```




    array([0.5, 1. , 1.5])




```python
#Take the sin of all values

np.sin(a)
```




    array([0.84147098, 0.90929743, 0.14112001])




```python
np.cos(a)
```




    array([ 0.54030231, -0.41614684, -0.9899925 ])



 ### Linear Algebra


```python
a=np.ones((2,3))
print(a)

b=np.full((3,2),2)
print(b)
```

    [[1. 1. 1.]
     [1. 1. 1.]]
    [[2 2]
     [2 2]
     [2 2]]
    


```python
a*b #Since both these matrices are of different so this direct multiplication will lead to errors.

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-16-8ce765dcfa30> in <module>
    ----> 1 a*b
    

    ValueError: operands could not be broadcast together with shapes (2,3) (3,2) 



```python
#Numpy has it's own matrice multiplication method

np.matmul(a,b)
```




    array([[6., 6.],
           [6., 6.]])




```python
#Find the determinant of the matrix
#The determinent of identity matrix is 1.

c=np.identity(3)
np.linalg.det(c)

```




    1.0



## Statistics



```python
stats=np.array([[1,2,3],[4,5,6]])
stats #Be very careful while putting on brackets.
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.min(stats) #This mill return the minimum element.
```




    1




```python
#Adding axis changes the answers and what we want them toc consider.

np.max(stats) #We get the answer as 6. #Looking at elements.

np.max(stats,axis=0) #We get the answer as ([4,5,6]) #Looking at rows

np.max(stats,axis=1) #We get the answer as ([3,6]) #Looking at columns
```




    array([3, 6])




```python
np.sum(stats)
```




    21




```python
np.sum(stats,axis=0) #Column wise addition
```




    array([5, 7, 9])




```python
np.sum(stats,axis=1) #row wise addition
```




    array([ 6, 15])



### Reorganizing Arrays


```python
#Reshaping arrays

before=np.array([[1,2,3,4],[5,6,7,8]])
print(before)


#You can reshape a matrix as long as all the elements of the previous matrix can fit into the new matrix shape.

after=before.reshape((4,2))
print(after)
```

    [[1 2 3 4]
     [5 6 7 8]]
    [[1 2]
     [3 4]
     [5 6]
     [7 8]]
    


```python
#Vertically stacking vectors

v1=np.array([1,2,3,4])
v2=np.array([5,6,7,8])

np.vstack([v1,v2]) #In the output we see that they are the part of the same matrix.
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
#We can use the above example for multiple rows of verticle stacking.

v1=np.array([1,2,3,4])
v2=np.array([5,6,7,8])
np.vstack([v1,v2,v1,v2])
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8],
           [1, 2, 3, 4],
           [5, 6, 7, 8]])




```python
#Working on horizontal stack

h1=np.ones((2,3))
h2=np.zeros((2,2))

np.hstack((h1,h2)) #Both square boxes and parenthesis work here.
```




    array([[1., 1., 1., 0., 0.],
           [1., 1., 1., 0., 0.]])



## Miscellaneous

### Load data from file


```python
filedata=np.genfromtxt('nameoffile',delimiter=',')
filedata

#The output that we will get will be more in terms of float data
filedata=filedata.astype('int32') #We are converting all the data into int

filedata
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-39-0a14bb57a065> in <module>
    ----> 1 filedata=np.genfromtxt('nameoffile',delimiter=',')
          2 filedata
          3 
          4 #The output that we will get will be more in terms of float data
          5 filedata=filedata.astype('int32') #We are converting all the data into int
    

    ~\anaconda3\lib\site-packages\numpy\lib\npyio.py in genfromtxt(fname, dtype, comments, delimiter, skip_header, skip_footer, converters, missing_values, filling_values, usecols, names, excludelist, deletechars, replace_space, autostrip, case_sensitive, defaultfmt, unpack, usemask, loose, invalid_raise, max_rows, encoding, like)
       1789             fname = os_fspath(fname)
       1790         if isinstance(fname, str):
    -> 1791             fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
       1792             fid_ctx = contextlib.closing(fid)
       1793         else:
    

    ~\anaconda3\lib\site-packages\numpy\lib\_datasource.py in open(path, mode, destpath, encoding, newline)
        192 
        193     ds = DataSource(destpath)
    --> 194     return ds.open(path, mode, encoding=encoding, newline=newline)
        195 
        196 
    

    ~\anaconda3\lib\site-packages\numpy\lib\_datasource.py in open(self, path, mode, encoding, newline)
        529                                       encoding=encoding, newline=newline)
        530         else:
    --> 531             raise IOError("%s not found." % path)
        532 
        533 
    

    OSError: nameoffile not found.


##Boolean Masking and Advanced Indexing


```python
filedata>50 #It will put a boolean mask over the values... i.e. for values greater than 50 it will represent that valie as True..and those not fulfilling the condition as false.
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-38-31a09eb2a376> in <module>
    ----> 1 filedata>50 #It will put a boolean mask over the values... i.e. for values greater than 50 it will represent that valie as True..and those not fulfilling the condition as false.
    

    NameError: name 'filedata' is not defined



```python
filedata[filedata>50] #Will return an array of values in the main matrix which is greater than 50.
```

## You can index with a list in a numpy


```python
u=np.array([1,2,3,4,5,6,7,8])
u[[1,2,5]]
#It returns us the value of elements located at that index 
```




    array([2, 3, 6])




```python
#We want to check if there is any value greater than 3

np.any(filedata>3,axis=0) #So it will look down on each column and return an array of it, returning True or False values.
```




    True




```python
#We can also provide a range

((filedata>50)&(filedata<100)) #So all values greater than 50 but less than 100.
```


```python
(~((filedata>50)&(filedata<100))) #Now this will act as a negation.
```
