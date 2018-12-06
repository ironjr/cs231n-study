import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  ############################################################################
  f = X @ W
  f -= np.max(f, axis=0)
  for k in range(X.shape[0]):
    softmax = np.exp(f[k, :])
    softmax /= np.sum(softmax)
    loss -= np.log(softmax[y[k]])
    for j in range(dW.shape[0]): # dimensions
      for i in range(dW.shape[1]): # classes
        dW[j, i] += (softmax[i] - (i is y[k])) * X[k, j]
  loss /= X.shape[0]
  dW /= X.shape[0]
  
  # Regularization term
  regL = 0.0
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      regL += W[i, j] * W[i, j]
  loss += regL * reg
  dW += 2 * W * reg

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X @ W
  f -= np.max(f, axis=0)
  expf = np.exp(f)
  softmax = expf / np.sum(expf, axis=1)[:, None] 
  loss = np.sum(np.log(np.diag(softmax[:, y]))) / -X.shape[0]
  loss += reg * np.sum(np.multiply(W, W))

  ybool = np.transpose(np.tile(y, (W.shape[1], 1))) == np.arange(W.shape[1])
  dW += np.transpose(X) @ (softmax - ybool)
  dW /= X.shape[0]
  dW += 2 * W * reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

