from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    num_train=X.shape[0]
    num_classes=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***
    for i in range(num_train):
        scores=np.dot(X[i],W)
        scores=np.exp(scores)
        loss+=-np.log(scores[y[i]]/np.sum(scores))
        for j in range(num_classes):
            if j==y[i]:
                continue
            dW[:,j]+=(X[i,:]*scores[j]/np.sum(scores))
        dW[:,y[i]]+=(-X[i,:]*(np.sum(scores)-scores[y[i]])/np.sum(scores))
              
    pass
    loss/=num_train
    loss+=reg*np.sum(W**2)
    dW/=num_train
    dW+=2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_train=X.shape[0]
    dW = np.zeros_like(W)
    scores=X.dot(W)
    scores=np.exp(scores)
    correct_scores=scores[np.arange(num_train),y]
    sums=np.sum(scores,axis=1)
    loss+=(np.sum(-np.log(correct_scores/sums)))
    loss/=num_train
    loss+=reg*np.sum(W**2)
    mask1=scores
    mask1[np.arange(num_train),y]=0
    sums2=np.sum(mask1,axis=1)
    mask1=mask1/sums[:,np.newaxis]
    mask2=np.zeros_like(scores)
    mask2[np.arange(num_train),y]=sums2
    mask2=mask2/sums[:,np.newaxis]
    dW+=np.dot(X.T,mask1)-np.dot(X.T,mask2)
    dW/=num_train
    dW+=2*reg*W
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW