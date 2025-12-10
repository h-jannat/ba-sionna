"""
User Equipment (UE) Controller with Adaptive RNN-Based Beam Selection

This module implements the UE-side controller for adaptive receive beamforming in
mmWave communication systems. The controller uses a recurrent neural network (RNN)
to process sequential sensing measurements and generate optimal receive combining
vectors at each step.

Key Features:
    - 2-layer GRU/LSTM RNN for temporal processing
    - Adaptive receive beam generation based on sensing history
    - Feedback message generation for BS final beam refinement
    - Learns to extract channel information from noisy measurements

Operation Flow:
    At each sensing step t:
    1. Input: received signal y_t (complex scalar), BS beam index x_t
    2. RNN processes: [Re(y_t), Im(y_t), x_t/NCB] → hidden state h_t
    3. Output layer generates: receive beam w_t (complex vector, NRX-dim)
    4. Feedback layer generates: feedback m_t (real vector, NFB-dim)
    5. Beam is normalized: w_t ← w_t / ||w_t||

Network Architecture:
    Input: 3 features [Re(y_t), Im(y_t), x_t_normalized]
    ↓
    2-layer GRU/LSTM (hidden_size per layer)
    ↓
    ├─→ Dense(2*NRX) → Beam Output (w_t)
    └─→ Dense(NFB) → Feedback Output (m_t)

Trainable via backpropagation through the entire sensing sequence,
optimizing the final beamforming gain.

References:
    Paper Section III.B: UE Controller Design
    "Two layers of gated recurrent units (GRU)"
"""

import tensorflow as tf
from utils import normalize_beam, complex_to_real_vector, real_to_complex_vector


class UEController(tf.keras.Model):
    """
    UE Controller with RNN for adaptive beam selection.
    
    At each sensing step t, the UE:
    1. Receives the signal y_t and beam index x_t from BS
    2. Updates its internal state using RNN
    3. Generates the receive combining vector w_t
    4. Optionally generates feedback for BS
    """
    
    def __init__(self,
                 num_antennas,
                 rnn_hidden_size=128,
                 rnn_type="GRU",
                 num_feedback=4,
                 codebook_size=8,
                 scheme='C3',
                 **kwargs):
        """
        Args:
            num_antennas: Number of receive antennas (NRX)
            rnn_hidden_size: Hidden state size for RNN
            rnn_type: Type of RNN ("GRU" or "LSTM")
            num_feedback: Number of real-valued feedback values (NFB)
            codebook_size: BS codebook size for beam index normalization
            scheme: Training scheme ('C1', 'C2', or 'C3')
        """
        super().__init__(**kwargs)
        self.num_antennas = num_antennas
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_type = rnn_type
        self.num_feedback = num_feedback
        self.codebook_size = codebook_size
        self.scheme = scheme
        
        # RNN layer - 2 layers as specified in paper
        if rnn_type == "GRU":
            # Paper: "two layers of gated recurrent units"
            # Use stacked GRU cells for proper 2-layer architecture
            cells = [
                tf.keras.layers.GRUCell(rnn_hidden_size, name=f'ue_gru_cell_layer{i+1}')
                for i in range(2)
            ]
            self.rnn = tf.keras.layers.RNN(
                cells,
                return_sequences=True,
                return_state=True,
                name='ue_2layer_gru'
            )
            self.num_layers = 2
        elif rnn_type == "LSTM":
            # Also support 2-layer LSTM for compatibility
            cells = [
                tf.keras.layers.LSTMCell(rnn_hidden_size, name=f'ue_lstm_cell_layer{i+1}')
                for i in range(2)
            ]
            self.rnn = tf.keras.layers.RNN(
                cells,
                return_sequences=True,
                return_state=True,
                name='ue_2layer_lstm'
            )
            self.num_layers = 2
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layers
        # Beam generation: map hidden state to complex beam
        # Output 2*NRX values (real and imaginary parts)
        self.beam_output = tf.keras.layers.Dense(
            2 * num_antennas,
            activation=None,
            name='beam_output'
        )
        
        # Feedback generation - scheme dependent
        # C1: Output beam index (NCB logits → argmax)
        # C2/C3: Output feedback vector (NFB values)
        if scheme == 'C1':
            # For C1: output logits over beam indices
            self.feedback_output = tf.keras.layers.Dense(
                codebook_size,  # NCB dimensional logits
                activation=None,
                name='feedback_beam_index_logits'
            )
        else:  # C2 or C3
            # For C2/C3: output feedback vector
            self.feedback_output = tf.keras.layers.Dense(
                num_feedback,
                activation=None,
                name='feedback_output'
            )
    
    def get_initial_state(self, batch_size):
        """
        Get initial hidden state for RNN.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial state(s) for RNN (list of states for 2-layer RNN)
        """
        if self.rnn_type == "GRU":
            # 2-layer GRU: return list of 2 hidden states
            return [
                tf.zeros([batch_size, self.rnn_hidden_size]),  # Layer 1
                tf.zeros([batch_size, self.rnn_hidden_size])   # Layer 2
            ]
        elif self.rnn_type == "LSTM":
            # 2-layer LSTM: return list of 2 [hidden, cell] state pairs
            return [
                [tf.zeros([batch_size, self.rnn_hidden_size]),  # Layer 1 hidden
                 tf.zeros([batch_size, self.rnn_hidden_size])], # Layer 1 cell
                [tf.zeros([batch_size, self.rnn_hidden_size]),  # Layer 2 hidden
                 tf.zeros([batch_size, self.rnn_hidden_size])]  # Layer 2 cell
            ]
    
    def process_step(self, received_signal, beam_index, state):
        """
        Process one sensing step.
        
        Args:
            received_signal: Received complex signal y_t, shape (batch,)
            beam_index: BS beam index x_t, shape (batch,)
            state: RNN hidden state from previous step
            
        Returns:
            combining_vector: Receive beam w_t, shape (batch, num_antennas)
            feedback: Feedback values, shape (batch, num_feedback) or (batch, 1) for C1
            new_state: Updated RNN state
            feedback_logits: For C1 only, beam logits (batch, ncb); None for C2/C3
        """
        batch_size = tf.shape(received_signal)[0]
        
        # Prepare RNN input
        # Concatenate: [real(y_t), imag(y_t), x_t (one-hot or scalar)]
        y_real = tf.reshape(tf.math.real(received_signal), [batch_size, 1])
        y_imag = tf.reshape(tf.math.imag(received_signal), [batch_size, 1])
        
        # Normalize beam index to [0, 1] range using actual codebook size
        x_normalized = tf.reshape(
            tf.cast(beam_index, tf.float32) / tf.cast(self.codebook_size - 1, tf.float32),
            [batch_size, 1]
        )
        
        # RNN input: (batch, 1, input_dim)
        rnn_input = tf.concat([y_real, y_imag, x_normalized], axis=-1)
        rnn_input = tf.expand_dims(rnn_input, axis=1)  # (batch, 1, 3)
        
        # Run RNN (2-layer)
        if self.rnn_type == "GRU":
            # For 2-layer GRU with stacked cells
            rnn_output, *new_states = self.rnn(rnn_input, initial_state=state)
            # new_states is list of 2 states [layer1_state, layer2_state]
        elif self.rnn_type == "LSTM":
            # For 2-layer LSTM with stacked cells
            outputs = self.rnn(rnn_input, initial_state=state)
            rnn_output = outputs[0]
            new_states = outputs[1:]  # List of state tuples

        
        # Remove time dimension: (batch, 1, hidden) -> (batch, hidden)
        rnn_output = tf.squeeze(rnn_output, axis=1)
        
        # Generate combining vector
        beam_real_imag = self.beam_output(rnn_output)  # (batch, 2*NRX)
        combining_vector = real_to_complex_vector(beam_real_imag, self.num_antennas)
        
        # Normalize beam
        combining_vector = normalize_beam(combining_vector)
        
        # Generate feedback (scheme-dependent)
        feedback_logits = None  # Only used for C1
        if self.scheme == 'C1':
            # C1: Generate beam index from logits
            beam_logits = self.feedback_output(rnn_output)  # (batch, NCB)
            # Convert to beam index (will be used by BS to pick from codebook)
            # Note: tf.argmax is non-differentiable, but this is expected for C1
            # The cross-entropy loss will train this layer despite the argmax
            feedback = tf.cast(tf.argmax(beam_logits, axis=-1), tf.float32)  # (batch,)
            feedback = tf.expand_dims(feedback, -1)  # (batch, 1) for consistency
            feedback_logits = beam_logits  # Return logits for CE loss computation
        else:  # C2 or C3
            # C2/C3: Generate feedback vector
            feedback = self.feedback_output(rnn_output)  # (batch, NFB)
        
        return combining_vector, feedback, new_states, feedback_logits
    
    def call(self, received_signals, beam_indices, initial_state=None):
        """
        Process a sequence of sensing steps.
        
        Args:
            received_signals: Sequence of received signals, shape (batch, T)
            beam_indices: Sequence of beam indices, shape (batch, T)
            initial_state: Initial RNN state (optional)
            
        Returns:
            combining_vectors: Sequence of receive beams, shape (batch, T, num_antennas)
            feedbacks: Sequence of feedback values, shape (batch, T, num_feedback)
        """
        batch_size = tf.shape(received_signals)[0]
        T = tf.shape(received_signals)[1]
        
        # Initialize state if not provided
        if initial_state is None:
            state = self.get_initial_state(batch_size)
        else:
            state = initial_state
        
        # Lists to store outputs
        combining_vectors_list = []
        feedbacks_list = []
        
        # Process each time step
        for t in range(T):
            y_t = received_signals[:, t]
            x_t = beam_indices[:, t]
            
            combining_vector, feedback, state = self.process_step(y_t, x_t, state)
            
            combining_vectors_list.append(combining_vector)
            feedbacks_list.append(feedback)
        
        # Stack outputs
        combining_vectors = tf.stack(combining_vectors_list, axis=1)  # (batch, T, NRX)
        feedbacks = tf.stack(feedbacks_list, axis=1)  # (batch, T, NFB)
        
        return combining_vectors, feedbacks


if __name__ == "__main__":
    print("Testing UE Controller...")
    print("=" * 60)
    
    # Create UE controller
    ue_controller = UEController(
        num_antennas=16,
        rnn_hidden_size=128,
        rnn_type="GRU",
        num_feedback=4
    )
    
    # Test single step processing
    batch_size = 10
    received_signal = tf.complex(
        tf.random.normal([batch_size]),
        tf.random.normal([batch_size])
    )
    beam_index = tf.constant([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    
    initial_state = ue_controller.get_initial_state(batch_size)
    print(f"Initial state shape: {initial_state.shape if isinstance(initial_state, tf.Tensor) else [s.shape for s in initial_state]}")
    
    combining_vector, feedback, new_state = ue_controller.process_step(
        received_signal, beam_index, initial_state
    )
    
    print(f"\nSingle step output:")
    print(f"  Combining vector shape: {combining_vector.shape}")
    print(f"  Combining vector norm: {tf.norm(combining_vector, axis=-1)[:3]}")
    print(f"  Feedback shape: {feedback.shape}")
    
    # Test sequence processing
    T = 8
    received_signals = tf.complex(
        tf.random.normal([batch_size, T]),
        tf.random.normal([batch_size, T])
    )
    beam_indices = tf.tile(tf.reshape(tf.range(T), [1, T]), [batch_size, 1])
    
    combining_vectors, feedbacks = ue_controller(received_signals, beam_indices)
    
    print(f"\nSequence processing:")
    print(f"  Combining vectors shape: {combining_vectors.shape}")
    print(f"  Feedbacks shape: {feedbacks.shape}")
    print(f"  Beam norms (first sample, all steps): {tf.norm(combining_vectors[0], axis=-1)}")
    
    # Test trainability
    print(f"\nNumber of trainable variables: {len(ue_controller.trainable_variables)}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
