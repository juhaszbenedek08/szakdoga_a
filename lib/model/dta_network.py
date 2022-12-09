from functools import cached_property

from torch.nn import Module, Sequential, Sigmoid

from lib.util.network_util import ParallelConnector, Concater, Rearranger, Single


class DTANetwork(Module):

    def __init__(
            self,
            drug_encoder: Module,
            drug_decoder: Module,
            target_encoder: Module,
            target_decoder: Module,
            predictor: Module
    ):
        super().__init__()
        self.drug_encoder = drug_encoder
        self.drug_decoder = drug_decoder
        self.target_encoder = target_encoder
        self.target_decoder = target_decoder
        self.predictor = predictor

    # @cached_property
    # def drug_autoencoder(self):
    #     return Sequential(self.drug_encoder, self.drug_decoder)

    # @cached_property
    # def target_autoencoder(self):
    #     return Sequential(self.target_encoder, self.target_decoder)

    # @cached_property
    # def offline_predictor(self):
    #     return Sequential(
    #         Offline(
    #             ParallelConnector(
    #                 self.drug_encoder,
    #                 self.target_encoder
    #             )
    #         ),
    #         Concater(),
    #         self.predictor
    #     )

    @cached_property
    def online_predictor_with_sigmoid(self):
        return Sequential(
            ParallelConnector(
                self.drug_encoder,
                self.target_encoder
            ),
            Concater(),
            self.predictor,
            Sigmoid()
        )

    @cached_property
    def combined_predictor(self):
        return Sequential(
            ParallelConnector(
                self.drug_encoder,
                self.target_encoder
            ),
            Rearranger(
                [0],
                [1],
                [0, 1],
                [0, 1]
            ),
            ParallelConnector(
                Sequential(
                    Single(),
                    self.drug_decoder
                ),
                Sequential(
                    Single(),
                    self.target_decoder
                ),
                Sequential(
                    Concater(),
                    self.predictor
                ),
                Concater(),
            )
        )
