import unittest

import numpy as np
import torch

from extract_dvcs_cff.inference.deepsets import DeepSetsDatasetEncoder
from extract_dvcs_cff.inference.stage10 import (
    STAGE10_GLOBAL_FEATURE_NAMES,
    STAGE10_POINT_FEATURE_NAMES,
    build_stage10_context,
)


class MaskedDesignTest(unittest.TestCase):
    def test_context_padding_has_zero_point_mask(self):
        points = [
            {
                "x_b": 0.2, "t_GeV2": -0.2, "Q2_GeV2": 2.0,
                "beam_energy_GeV": 10.6, "phi_rad": float(index),
                "observable_id": "DVCSCrossSectionUUMinus",
            }
            for index in range(2)
        ]
        context = build_stage10_context(
            points=points,
            observed=np.asarray((0.1, 0.2)),
            covariance=np.eye(2),
            global_normalization_response=np.ones(2),
            lu_normalization_response=np.zeros(2),
            maximum_point_count=3,
        )
        width = len(STAGE10_POINT_FEATURE_NAMES)
        tokens = context[:3 * width].reshape(3, width)
        np.testing.assert_array_equal(tokens[:, -1], (1.0, 1.0, 0.0))
        np.testing.assert_array_equal(tokens[2], np.zeros(width))

    def test_all_active_pool_matches_historical_mean_exactly(self):
        torch.manual_seed(7)
        encoder = DeepSetsDatasetEncoder(
            token_count=3, feature_count=4, global_feature_count=2,
            point_hidden=5, embedding_features=3,
        )
        context = torch.randn(4, 14)
        tokens = context[:, :12].reshape(4, 3, 4)
        tokens[:, :, -1] = 1.0
        context = torch.cat((tokens.reshape(4, 12), context[:, 12:]), dim=1)
        historical = encoder.dataset_encoder(torch.cat((
            encoder.point_encoder(tokens).mean(dim=1), context[:, 12:],
        ), dim=1))
        self.assertTrue(torch.equal(encoder(context), historical))

    def test_masked_padding_cannot_create_bias_pseudodata(self):
        torch.manual_seed(11)
        encoder = DeepSetsDatasetEncoder(
            token_count=3, feature_count=4, point_hidden=5,
            embedding_features=3,
        )
        left = torch.randn(2, 12)
        tokens = left.reshape(2, 3, 4)
        tokens[:, :, -1] = 1.0
        tokens[:, 2, -1] = 0.0
        left = tokens.reshape(2, 12)
        right = left.clone()
        right.reshape(2, 3, 4)[:, 2, :-1] = 1.0e6
        self.assertTrue(torch.equal(encoder(left), encoder(right)))


if __name__ == "__main__":
    unittest.main()
