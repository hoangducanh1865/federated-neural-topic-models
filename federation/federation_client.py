class FederationClient:
    """Class for representing a client that has connected to the server for the federated learning of 
       a CTM or ProdLDA model.
    """

    def __init__(self, id, federation_key, tensor, current_epoch, id_request, num_max_epochs):
        self.id = id
        self.federation_key = federation_key
        self.tensor = tensor
        self.current_epoch = current_epoch
        self.id_request = id_request # TODO: revise this field; not really necessary
        self.num_max_epochs = num_max_epochs

    def get_pos_by_key(key, federation_clients):
        """It searchs a client with the specified key over the given list of FederationClient objects
           and when found, it returns its position in the list.
        Args:
            * key (int): Dictionary id of the client 
            * federation_clients (FederationClient): List of FederationClient objects representing the clients
                                                   that are connected to the client.

        Returns:
            * int: Position on the list of the searched client
        """
        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                return client_pos
            else:
                print("No client with specified key was found")
                return -1

    def update_client_state(self, tensor, current_epoch, id_request):
        self.tensor = tensor
        self.current_epoch = current_epoch, 
        self.id_request = id_request