from abc import ABC
from typing import List, Sequence, Optional

import attr
import torch
from attr import asdict, define, field

import esm.utils.constants.api as C
from esm.tokenization import (
    TokenizerCollectionProtocol,
    get_esm3_model_tokenizers,
)
from esm.utils import encoding
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.misc import (
    get_chainbreak_boundaries_from_sequence,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.types import FunctionAnnotation, PathOrBuffer
from esm.sdk.api import ESMProtein, ESMProteinTensor

from esm.models.esmc import ESMC

import numpy as np


def stack_variable_length_tensors(
    sequences: Sequence[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Automatically stack tensors together, padding variable lengths with the
    value in constant_value. Handles an arbitrary number of dimensions.

    Examples:
        >>> tensor1, tensor2 = torch.ones([2]), torch.ones([5])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5]. First row is [1, 1, 0, 0, 0]. Second row is all ones.

        >>> tensor1, tensor2 = torch.ones([2, 4]), torch.ones([5, 3])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5, 4]
    """
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype
    device = sequences[0].device

    array = torch.full(shape, constant_value, dtype=dtype, device=device)
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


@define
class BatchedESMProtein:
    """
    Batched version of ESMProtein for efficient batch processing.
    Handles multiple proteins as a single batch.
    """

    sequences: list[str | None]
    secondary_structures: list[str | None] = field(factory=list)
    sasa: list[list[float | None] | None] = field(factory=list)
    function_annotations: list[list[FunctionAnnotation] | None] = field(factory=list)
    coordinates: list[torch.Tensor | None] = field(factory=list)

    # Metrics
    plddts: list[torch.Tensor | None] = field(factory=list)
    ptms: list[torch.Tensor | None] = field(factory=list)

    # Safety flags
    potential_sequence_of_concern: list[bool] = field(factory=list)

    @classmethod
    def from_proteins(cls, proteins: Sequence["ESMProtein"]) -> "BatchedESMProtein":
        """Create a BatchedESMProtein from a sequence of individual ESMProteins."""
        return cls(
            sequences=[p.sequence for p in proteins],
            secondary_structures=[p.secondary_structure for p in proteins],
            sasa=[p.sasa for p in proteins],
            function_annotations=[p.function_annotations for p in proteins],
            coordinates=[p.coordinates for p in proteins],
            plddts=[p.plddt for p in proteins],
            ptms=[p.ptm for p in proteins],
            potential_sequence_of_concern=[
                p.potential_sequence_of_concern for p in proteins
            ],
        )

    def to_proteins(self) -> list[ESMProtein]:
        """Convert batched representation back to list of individual ESMProteins."""
        proteins = []
        for i in range(len(self.sequences)):
            proteins.append(
                ESMProtein(
                    sequence=self.sequences[i],
                    secondary_structure=self.secondary_structures[i]
                    if self.secondary_structures
                    else None,
                    sasa=self.sasa[i] if self.sasa else None,
                    function_annotations=self.function_annotations[i]
                    if self.function_annotations
                    else None,
                    coordinates=self.coordinates[i] if self.coordinates else None,
                    plddt=self.plddts[i] if self.plddts else None,
                    ptm=self.ptms[i] if self.ptms else None,
                    potential_sequence_of_concern=self.potential_sequence_of_concern[i]
                    if self.potential_sequence_of_concern
                    else False,
                )
            )
        return proteins

    def __len__(self) -> int:
        """Return the batch size."""
        return len(self.sequences)


@define
class BatchedESMProteinTensor:
    """
    Batched version of ESMProteinTensor for efficient batch processing.
    All tensors should have matching batch dimensions.
    """

    sequence: Optional[torch.Tensor] = None
    structure: Optional[torch.Tensor] = None
    secondary_structure: Optional[torch.Tensor] = None
    sasa: Optional[torch.Tensor] = None
    function: Optional[torch.Tensor] = None
    residue_annotations: Optional[torch.Tensor] = None
    coordinates: Optional[torch.Tensor] = None

    # Safety flags as a batch
    potential_sequence_of_concern: torch.Tensor = field(
        factory=lambda: torch.zeros(0, dtype=torch.bool)
    )

    @classmethod
    def from_protein_tensors(
        cls, tensors: Sequence["ESMProteinTensor"]
    ) -> "BatchedESMProteinTensor":
        """Create a BatchedESMProteinTensor from a sequence of individual ESMProteinTensors."""

        def stack_tensors(getter):
            tensors_to_stack = [getter(t) for t in tensors if getter(t) is not None]
            return torch.stack(tensors_to_stack) if tensors_to_stack else None

        return cls(
            sequence=stack_tensors(lambda t: t.sequence),
            structure=stack_tensors(lambda t: t.structure),
            secondary_structure=stack_tensors(lambda t: t.secondary_structure),
            sasa=stack_tensors(lambda t: t.sasa),
            function=stack_tensors(lambda t: t.function),
            residue_annotations=stack_tensors(lambda t: t.residue_annotations),
            coordinates=stack_tensors(lambda t: t.coordinates),
            potential_sequence_of_concern=torch.tensor(
                [t.potential_sequence_of_concern for t in tensors], dtype=torch.bool
            ),
        )

    def to_protein_tensors(self) -> list[ESMProteinTensor]:
        """Convert batched tensor representation back to list of individual ESMProteinTensors."""
        batch_size = self._get_batch_size()
        if batch_size == 0:
            return []

        tensors = []
        for i in range(batch_size):
            tensors.append(
                ESMProteinTensor(
                    sequence=self.sequence[i] if self.sequence is not None else None,
                    structure=self.structure[i] if self.structure is not None else None,
                    secondary_structure=self.secondary_structure[i]
                    if self.secondary_structure is not None
                    else None,
                    sasa=self.sasa[i] if self.sasa is not None else None,
                    function=self.function[i] if self.function is not None else None,
                    residue_annotations=self.residue_annotations[i]
                    if self.residue_annotations is not None
                    else None,
                    coordinates=self.coordinates[i]
                    if self.coordinates is not None
                    else None,
                    potential_sequence_of_concern=bool(
                        self.potential_sequence_of_concern[i]
                    ),
                )
            )
        return tensors

    def _get_batch_size(self) -> int:
        """Determine batch size from the first non-None tensor attribute."""
        for tensor in [
            self.sequence,
            self.structure,
            self.secondary_structure,
            self.sasa,
            self.function,
            self.residue_annotations,
            self.coordinates,
        ]:
            if tensor is not None:
                return tensor.size(0)
        return len(self.potential_sequence_of_concern)

    def __len__(self) -> int:
        """Return the batch size."""
        return self._get_batch_size()

    @property
    def device(self) -> torch.device:
        """Get the device of the tensors."""
        for tensor in [
            self.sequence,
            self.structure,
            self.secondary_structure,
            self.sasa,
            self.function,
            self.residue_annotations,
            self.coordinates,
        ]:
            if tensor is not None:
                return tensor.device
        return self.potential_sequence_of_concern.device

    def to(
        self, device_or_dtype: str | torch.device | torch.dtype
    ) -> "BatchedESMProteinTensor":
        """Move all tensors to the specified device or cast to dtype."""

        def move_tensor(tensor):
            return tensor.to(device_or_dtype) if tensor is not None else None

        return BatchedESMProteinTensor(
            sequence=move_tensor(self.sequence),
            structure=move_tensor(self.structure),
            secondary_structure=move_tensor(self.secondary_structure),
            sasa=move_tensor(self.sasa),
            function=move_tensor(self.function),
            residue_annotations=move_tensor(self.residue_annotations),
            coordinates=move_tensor(self.coordinates),
            potential_sequence_of_concern=self.potential_sequence_of_concern.to(
                device_or_dtype
            ),
        )

    @classmethod
    def empty(
        cls,
        batch_size: int,
        seq_length: int,
        tokenizers: Optional[TokenizerCollectionProtocol] = None,
        device: str | torch.device = "cpu",
    ) -> "BatchedESMProteinTensor":
        """Create an empty batched tensor with given batch size and sequence length."""
        if tokenizers is None:
            tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)

        def batch_default_tokens(get_default_fn):
            tokens = get_default_fn(seq_length, tokenizers)
            return torch.stack([tokens] * batch_size).to(device)

        return cls(
            sequence=batch_default_tokens(encoding.get_default_sequence_tokens),
            structure=batch_default_tokens(encoding.get_default_structure_tokens),
            secondary_structure=batch_default_tokens(
                encoding.get_default_secondary_structure_tokens
            ),
            sasa=batch_default_tokens(encoding.get_default_sasa_tokens),
            function=batch_default_tokens(encoding.get_default_function_tokens),
            residue_annotations=batch_default_tokens(
                encoding.get_default_residue_annotation_tokens
            ),
            potential_sequence_of_concern=torch.zeros(batch_size, dtype=torch.bool).to(
                device
            ),
        )


class BatchedESMC(ESMC):
    """
    Batched version of ESMC for efficient batch processing.
    Handles multiple ESMC models as a single batch.
    """

    def _batch_tokenize(self, sequences: list[list[str]]) -> torch.Tensor:
        pad = self.tokenizer.pad_token_id
        assert pad is not None

        # Tokenize all sequences in the batch
        tokenized_sequences = [
            encoding.tokenize_sequence(seq, self.tokenizer, add_special_tokens=True)
            for seq in sequences
        ]

        # Stack and pad sequences to the same length
        batched_tokens = stack_variable_length_tensors(
            tokenized_sequences,
            constant_value=pad,
        ).to(next(self.parameters()).device)

        return batched_tokens

    def batch_encode(self, input_protein: BatchedESMProtein) -> BatchedESMProteinTensor:
        """
        Encode a batched protein representation into a batched tensor representation.

        Args:
            input (BatchedESMProtein): Batched protein representation to encode

        Returns:
            BatchedESMProteinTensor: Batched tensor representation
        """
        # Make a copy to avoid modifying original
        input_protein = attr.evolve(input_protein)

        # Check sequences exist and convert
        sequences = []
        for sequence in input_protein.sequences:
            if sequence is None:
                raise ValueError("All proteins in batch must have sequences to encode")
            sequences.append(sequence)

        # Batch tokenize
        sequence_tokens = self._batch_tokenize(sequences)

        # Handle coordinates only if all are present and non-None
        coordinates = None
        if hasattr(input_protein, "coordinates") and input_protein.coordinates:
            if all(c is not None for c in input_protein.coordinates):
                coordinates = torch.stack(input_protein.coordinates)

        # Create batched tensor
        return BatchedESMProteinTensor(
            sequence=sequence_tokens,
            coordinates=coordinates,
            potential_sequence_of_concern=torch.tensor(
                input_protein.potential_sequence_of_concern
                if input_protein.potential_sequence_of_concern
                else [False] * len(sequences),
                dtype=torch.bool,
                device=self.device,
            ),
        ).to(self.device)

    def batch_decode(self, input_tensor: BatchedESMProteinTensor) -> BatchedESMProtein:
        """
        Decode a batched tensor representation back to batched protein representation.

        Args:
            input (BatchedESMProteinTensor): Batched tensor to decode

        Returns:
            BatchedESMProtein: Batched protein representation
        """
        input_tensor = attr.evolve(input_tensor)  # Make a copy

        if input_tensor.sequence is None:
            raise ValueError("Tensor must have sequence to decode")

        # Decode sequences
        sequences = self._detokenize(input_tensor.sequence)

        # Convert coordinates if present
        coordinates = None
        if input_tensor.coordinates is not None:
            coordinates = [coord for coord in input_tensor.coordinates]

        # Create BatchedESMProtein with all available information
        return BatchedESMProtein(
            sequences=sequences,
            coordinates=coordinates,
            potential_sequence_of_concern=[
                bool(x) for x in input_tensor.potential_sequence_of_concern.tolist()
            ],
        )

    @classmethod
    def from_pretrained(
        cls, model_name: str = "ESMC_300M", device: torch.device | None = None
    ) -> "BatchedESMC":
        from esm.pretrained import load_local_model

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_local_model(model_name, device=device)
        if device.type != "cpu":
            model = model.to(torch.bfloat16)

        # Convert ESMC model to BatchedESMC if needed
        if isinstance(model, ESMC) and not isinstance(model, BatchedESMC):
            # Create a new BatchedESMC instance with the same parameters
            batched_model = cls.__new__(cls)
            # Copy all attributes from the original model
            batched_model.__dict__.update(model.__dict__)
            model = batched_model

        assert isinstance(model, BatchedESMC)
        return model