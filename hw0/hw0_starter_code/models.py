"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        self.fact_U = self.regr_U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.fact_Q = self.regr_Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        # Biases for users and items
        self.fact_A = self.regr_A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.fact_B = self.regr_B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # Embeddings for regression and factorization shared if embedding_sharing=True
        if not embedding_sharing:
            self.regr_U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
            self.regr_Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)
            self.regr_A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim)
            self.regr_B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim)

        # Define the MLP
        self.linear1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.activation = nn.ReLU()
        # Output the predicted score (size 1)
        self.linear2 = nn.Linear(layer_sizes[1], 1)

        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
        # Compute embeddings for users and items
        user_embedding = self.fact_U(user_ids)
        item_embedding = self.fact_Q(item_ids)
        user_bias = self.fact_A(user_ids)
        item_bias = self.fact_B(item_ids)

        predictions = (user_embedding * item_embedding + user_bias + item_bias).sum(1)

        interaction = torch.cat([user_embedding, item_embedding, user_embedding*item_embedding], dim=1)

        x = self.linear1(interaction)
        x = self.activation(x)
        score = self.linear2(x).squeeze()

        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score