#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Import the numpy library (01)
import numpy as np
print (np.__version__)


# In[6]:


#Define the 1D array
arr1 = np.array([0,1,2,3,4,5,6,7,8,9])
print(arr1)


# In[18]:


#Define the 2D array (02)
arr2 = np.array([[11,12,13,14],
               [21,22,23,24],
                [31,32,33,34]])
print (arr2.ndim)
print (arr2.shape)
print (arr2)


# In[26]:


#Getting the odd number from an array (03)
arr = np.arange(10) 
arr[arr%2 ==1]


# In[27]:


# Replace all odd numbers with -1 (04)
arr = np.arange(10)
arr[arr % 2 == 1] = -1
arr


# In[33]:


#Covert 1D array into 2D array with 3 rows and 3 columns (05)
arr = np.arange(9)
arr.reshape (3,3)


# In[36]:


#Stack two array vertically (06)
a = np.arange(9).reshape(3,3)
b = np.repeat(3,9).reshape(3,3)
np.vstack([a,b])


# In[37]:


#Get the common items from two arrays (07)
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# In[38]:


# Remove items from one array those exist in another array (08)
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# In[40]:


# Get the position where elements of two arrays match (09)
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a == b)


# In[44]:


# Extract all items from array between range (all items between 5 and 10)(10)
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a >=5) & (a <=10))
a[index]


# In[55]:


# How to swap two columns in a 2d numpy array? (11)
arr = np.arange(9).reshape(3,3) # swap column 1 and 2 in arr
arr[:,[1,0,2]]


# In[65]:


#How to swap two rows in a 2d numpy array? (12)
arr = np.arange(9).reshape(3,3) # Reverse the rows of a 2D array
arr[::-1]


# In[74]:


# How to reverse the columns of a 2D array? (13)
arr = np.arange(9).reshape(3,3)
arr[:, ::-1]


# In[78]:


# How to create a 2D array containing random floats between 5 and 10? (14)
arr = np.arange(9).reshape(3,3)
rand_arr = np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
print(rand_arr)


# In[82]:


# How to print only 3 decimal places in python numpy array? (15)
rand_arr = np.random.random([5,3])
np.set_printoptions(precision=3)
rand_arr[:4]


# In[91]:


# print a numpy array by suppressing the scientific notation (16)
np.set_printoptions(suppress=False)
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
np.set_printoptions(suppress=True, precision=6) 
rand_arr


# In[101]:


# Limit the number of items printed in output of numpy array (17)
np.set_printoptions(threshold=6)
a = np.arange(20) # limit the output to maximum 6 items 
a


# In[109]:


# print the full numpy array without truncating (18)
np.set_printoptions(threshold = np.nan)
a = np.arange(20)
a


# In[112]:


# Make a python function that handles scalars to work on numpy arrays (19)
def maxx(x, y):
   
    if x >= y:
        return x
    else:
        return y
pair_max = np.vectorize(maxx, otypes=[float])
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
pair_max(a, b)


# In[115]:


# Generate custom sequences in numpy without hardcoding (20)
a = np.array([1,2,3])
np.r_[np.repeat(a,2),np.tile(a,2)]


# In[135]:


# around function (21)
a = np.array([1.0001,2.0002,3.0003,4.0004,5.0005,6.0006,7.0007,8.0008,9.0009])
b = np.around(a,3)
b


# In[136]:


# sum function (22)
a = np.arange(9)
np.sum(a)


# In[147]:


# diff function, The first difference is given by out[i] = a[i+1] - a[i] along the given axis  (23)
x = np.array([1,2,4,7,0]) 
y = np.diff(x, n=2) # n int, optional. The number of times values are differenced. If zero, the input is returned as-is.
print(y)
np.diff(x, n=0)


# In[161]:


#  prod function (24)
x = np.array([2,2,4])
np.prod(x)


# In[166]:


# add function (25)
x = np.arange(15)
y = np.arange(15)
np.add(x,y)


# In[175]:


# positive function (26)
a = np.array([1,2,3,4,5,-6])
np.positive(a)


# In[176]:


# negative function (27)
a = np.array([1,2,3,4,5,-6])
np.negative(a)


# In[177]:


# multiply function (28)
a = np.arange(10)
np.multiply(a,2)


# In[184]:


# divide function (29)
a = np.array([2,4,6,8,10,12,14,16,18,20,21])
np.divide(a,2)


# In[186]:


# substract function (30)
x = np.array([2,4,6,8,10])
y = np.array([1,3,5,7,9])
np.subtract(x,y)


# In[188]:


# remainder function (31)
x = np.array([3,5,7,9,10])
np.remainder(x,2)


# In[196]:


# ndim function (32)
# shape function (33)
# size function (34)
# dtype function (35)
# itemsize function (36) the size in bytes of each element of the array
ndarray = np.array (
                    [
                       [ [111, 112, 113, 114], [121, 122, 123, 124] ,[131, 132, 133, 134] ],
                       [ [211, 212, 213, 214], [221, 222, 223, 224], [231, 232, 233, 234] ],
                       [ [311, 312, 313, 314], [321, 322, 323, 324], [331, 332, 333, 334] ],
                       [ [411, 412, 413, 414], [421, 422, 423, 424], [431, 432, 433, 434] ]
                    ]
                   )

print (ndarray.ndim)
print (ndarray.shape)
print (ndarray.size)
print (ndarray.dtype)
print (ndarray.itemsize)


# In[197]:


# zeros function (37) creating an array with n-dimensions all having zero values
np.zeros((3, 4))


# In[203]:


# ones function (38) creating an array with n-dimensions all having 1 as values
np.ones((2, 3, 4), dtype=np.int16)


# In[206]:


# min function (39)
# max function (40)
x = np.arange(15)
y = np.array([4,8,16,32,64,128])
print(np.min(x))
print(np.max(y))


# In[214]:


# reshape function (41)
b = np.arange(12).reshape(3,4)
b


# In[222]:


# one dimensional: Indexing and Slicing (42)
# (43)
# (44)
# (45)
a = np.arange(10)**3
print(a)
print(a[2])
print (a[4:7])
print (a[:5])
print (a[3:])


# In[231]:


# Copy function (46)
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a            # no new object is created. a and b are two names for the same ndarray object

def f(x):
     print(id(x))
id(a)
f(a)


# In[233]:


# view function, different array objects can share the same data (47)
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

c = a.view()
c


# In[234]:


# copy function (48)
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
d = a.copy()
d


# In[236]:


# clip function (49)
a = np.arange(10)
np.clip(a, 1,8)


# In[238]:


# absolute function (50)
x = np.array([1.2,-1.2])
np.absolute(x)

