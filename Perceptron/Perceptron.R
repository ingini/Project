data = data.frame( x1 = c( 0, 0, 1, 1 ),
                   x2 = c( 0, 1, 0, 1 ),
                   y =  c( 0, 1, 1, 1 ) )

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
