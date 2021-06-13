import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.signal import convolve2d
from pyensae.languages import r2python

data = {'x1':[0,0,1,1],'x2':[0,1,0,1],'y':[0,1,1,1]}
data = pd.DataFrame(data)

# 선 그래프 그리기
fig = plt.figure(figsize=(8,8)) # 캔버스 생성
fig.set_facecolor('white') # 캔버스 색상설정

# 선 그래프 생성
plt.scatter(data['x1'],data['x2'], c=data['y'],cmap='Blues',edgecolors='black',linewidths=2)
plt.colorbar()
plt.title('Dataset OR')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

 """data = data.frame( x1 = c( 0, 0, 1, 1 ),
                   x2 = c( 0, 1, 0, 1 ),
                   y =  c( 0, 1, 1, 1 ) )
#rscript =
print( data )

plot( data$x1, data$x2, type = 'n', main = 'Dataset OR', xlab = "x1", ylab = "x2" )
text( data$x1, data$x2, labels = data$y )
grid( nx = length( data$x1 ) + 1, ny = length( data$x1 ), col = 'black' )

weights = rnorm( mean = 0, sd = 0.1, n = ncol( data ) )

print( weights )

activation_function = function( net ){
  if( net > 0.5 )
    return( 1 )
  return( 0 )
}

data = as.matrix( data )

net = c( data[1, 1:2 ], 1 ) %*% weightss

y_hat = activation_function( net )

error = y_hat - data[1,3]

eta =  0.1

weights = weights - eta * ( error ) * c( data[ 1, 1:2 ], 1 )

print( weights )

perceptron = function( dataset, eta = 0.1, threshold = 1e-5 ){
  data = as.matrix( dataset )
  num.features = ncol( data ) - 1
  target = ncol( data )
  
  # Initial random  weights
  weights = rnorm( mean = 0, sd = 0.1, n = ncol( data ) )
  
  mse = threshold * 2
  while( mse > threshold ){
    mse = 0
    for( i in 1:nrow( data ) ){
      # Add bias and compute multiplications
      net = c( data[ i, 1:num.features ], 1 ) %*% weights
      
      # Activation function
      y_hat = activation_function( net )
      
      # Compute mse
      error = ( y_hat - data[ i, target ] )
      mse = mse + error^2
      cat( paste( "Mean square error = ", mse, "\n" ) )
      
      # Update weights
      weights = weights - eta * error * c( data[i, 1:num.features ], 1 )
    }
  }
  return( weights )
}


shattering.plane = function( weights ){
  X = seq( 0, 1, length = 100 )
  data = outer( X, X, function( X, Y ){ cbind( X, Y, 1 ) %*% weights } )
  id = which( data > 0.5 )
  data[ id ] = 1
  data[ -id ]= 0
  filled.contour( data )
}

weights = perceptron( data, eta=0.1, threshold=1e-5 )

shattering.plane( weights )
"""
# print(r2python(rscript, pep8=True))

weights = np.random.normal(0, 0.1, len(data)-1)
print(weights)

def activation_function(net):
    if net > 0.5:
        return 1
    return 0

# data = np.asmatrix(data)
c_data = np.array([data['x1'][0],data['x2'][0],1])
net =  np.dot(c_data,weights)
y_hat = activation_function(net)

error = y_hat - data['y'][0]
eta = 0.1
weights = weights - eta * (error) * np.array([data['x1'][0],data['x2'][0], 1])
print(weights)

def perceptron(dataset, eta=0.1, threshold=1e-5):
    data = np.matrix(dataset)
    num_features = len(data) - 2
    target = len(data)-1
    # Initial random  weights
    weights = np.random.normal(0, 0.1, len(data) - 1)
    mse = threshold * 2
    while mse > threshold:
        mse = 0
        for i in range(len(data)):
            # Add bias and compute multiplications
            #net = make_tuple(data[i, range(1, num_features)], 1) @ weights
            for j in range(len(data)):
                net = convolve2d(data['x1'][i],data['x2'][j],1)
                # Activation function
                y_hat = activation_function(net)
            # Compute mse
                error = (y_hat - data[i, target])
                mse = mse + error ** 2
                print("Mean square error = ", mse, "                    "))
            # Update weights
            weights = weights - eta * error * \
                make_tuple(data[i, range(1, num_features)], 1)
    return weights

def shattering_plane(weights):
    X = seq(0, 1, length=100)
    def dataouter(X, X, function(X, Y):
                  cbind(X, Y, 1) @ weights)
    id = which(data > 0.5)
    data[id] = 1
    data[- id] = 0
    filled_contour(data)

weights = perceptron(data, eta=0.1, threshold=1e-5)
shattering_plane(weights)