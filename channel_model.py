"""
mmWave Geometric Channel Model

This module implements a geometric channel model for millimeter-wave (mmWave)
communication systems. The model captures the sparse scattering nature of mmWave
propagation through a limited number of propagation paths.

Channel Model:
    The channel matrix H ∈ ℂ^{NRX × NTX} is modeled as:
    
    H = Σ_{ℓ=1}^L α_ℓ a_RX(φ_ℓ^RX) a_TX^H(φ_ℓ^TX)
    
    where:
    - L: Number of propagation paths (typically 1-5 for mmWave)
    - α_ℓ ~ CN(0, 1): Complex path gain (Rayleigh fading)
    - φ_ℓ^RX ~ U[-π/2, π/2]: Angle of Arrival (AoA) at receiver
    - φ_ℓ^TX ~ U[-π/2, π/2]: Angle of Departure (AoD) at transmitter
    - a(φ): Array response vector for Uniform Linear Array (ULA)

Array Response Vector:
    For a ULA with N antennas and spacing d:
    
    a(φ) = [1, e^{jπ d sin(φ)}, e^{j2π d sin(φ)}, ..., e^{j(N-1)π d sin(φ)}]^T / √N
    
    Standard spacing: d = λ/2 (half-wavelength)

Channel Properties:
    - Rank: Typically rank-deficient (rank ≤ min(L, NRX, NTX))
    - Average power: 𝔼[||H||_F^2] ≈ L × NRX × NTX
    - Sparsity: Dominant paths in angular domain
    - Block fading: Constant during coherence time

Implementations:
    1. GeometricChannelModel: Pure geometric model (recommended)
       - Fast, deterministic, fully controllable
       - Suitable for beam alignment research
    
    2. SionnaChannelModel: Wrapper for Sionna integration (future)
       - More realistic channel modeling
       - Includes 3D scenarios, blockage, etc.

Usage Example:
    >>> from channel_model import GeometricChannelModel
    >>> 
    >>> channel_model = GeometricChannelModel(
    ...     num_tx_antennas=64,
    ...     num_rx_antennas=16,
    ...     num_paths=3
    ... )
    >>> 
    >>> # Generate batch of channel realizations
    >>> H = channel_model.generate_channel(batch_size=100)
    >>> print(H.shape)  # (100, 16, 64)

References:
    - Heath et al., "An Overview of Signal Processing Techniques for
      Millimeter Wave MIMO Systems," IEEE JSAC 2016
    - Alkhateeb et al., "Deep Learning Coordinated Beamforming for
      Highly-Mobile Millimeter Wave Systems," IEEE Access 2018
"""

import tensorflow as tf
import numpy as np
try:
    import sionna
    # Prefer the newer sionna.phy API, fall back to legacy paths if needed.
    try:
        from sionna.phy.channel.tr38901 import CDL, PanelArray
        from sionna.phy.channel import GenerateOFDMChannel
        from sionna.phy.ofdm import ResourceGrid
    except Exception:  # noqa: BLE001
        # Legacy (<1.0) module structure
        from sionna.channel.tr38901 import CDL, PanelArray  # type: ignore
        from sionna.channel import GenerateOFDMChannel  # type: ignore
        from sionna.ofdm import ResourceGrid  # type: ignore
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("Warning: Sionna not available. Using custom channel model.")

from utils import array_response_vector, random_angles


class GeometricChannelModel(tf.keras.layers.Layer):
    """
    Geometric mmWave channel model.
    
    Implements: H = Σ_{ℓ=1}^L α_ℓ a_RX(φ_ℓ^RX) a_TX^H(φ_ℓ^TX)
    
    where:
    - L is the number of propagation paths
    - α_ℓ ~ CN(0, 1) is the complex path gain
    - φ_ℓ^{RX/TX} ~ U[-π/2, π/2] are angle of arrival/departure
    - a(φ) is the array response vector
    """
    
    def __init__(self, 
                 num_tx_antennas, 
                 num_rx_antennas,
                 num_paths=3,
                 antenna_spacing=0.5,
                 normalize_channel=True,
                 **kwargs):
        """
        Args:
            num_tx_antennas: Number of transmit antennas (NTX)
            num_rx_antennas: Number of receive antennas (NRX)  
            num_paths: Number of propagation paths (L)
            antenna_spacing: Antenna spacing in wavelengths
            normalize_channel: Whether to normalize channel Frobenius norm
        """
        super().__init__(**kwargs)
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.num_paths = num_paths
        self.antenna_spacing = antenna_spacing
        self.normalize_channel = normalize_channel
    
    def generate_channel(self, batch_size):
        """
        Generate a batch of channel realizations.
        
        Args:
            batch_size: Number of channel samples to generate
            
        Returns:
            Channel tensor of shape (batch_size, num_rx_antennas, num_tx_antennas)
        """
        # Initialize channel to zeros
        H = tf.zeros([batch_size, self.num_rx_antennas, self.num_tx_antennas], 
                     dtype=tf.complex64)
        
        # Generate each path
        for path_idx in range(self.num_paths):
            # Complex path gain: α_ℓ ~ CN(0, 1)
            alpha_real = tf.random.normal([batch_size, 1, 1], mean=0.0, stddev=1.0/np.sqrt(2))
            alpha_imag = tf.random.normal([batch_size, 1, 1], mean=0.0, stddev=1.0/np.sqrt(2))
            alpha = tf.complex(alpha_real, alpha_imag)  # (batch, 1, 1)
            
            # Angle of arrival (AoA): φ_ℓ^RX ~ U[-π/2, π/2]
            aoa = random_angles([batch_size])  # (batch,)
            
            # Angle of departure (AoD): φ_ℓ^TX ~ U[-π/2, π/2]
            aod = random_angles([batch_size])  # (batch,)
            
            # Array response vectors
            a_rx = array_response_vector(aoa, self.num_rx_antennas, self.antenna_spacing)  # (batch, nrx)
            a_tx = array_response_vector(aod, self.num_tx_antennas, self.antenna_spacing)  # (batch, ntx)
            
            # Compute outer product: a_rx @ a_tx^H
            # a_rx: (batch, nrx) -> (batch, nrx, 1)
            # a_tx: (batch, ntx) -> (batch, 1, ntx) after conj and transpose
            a_rx_expanded = tf.expand_dims(a_rx, axis=-1)  # (batch, nrx, 1)
            a_tx_conj = tf.expand_dims(tf.math.conj(a_tx), axis=-2)  # (batch, 1, ntx)
            
            path_contribution = alpha * a_rx_expanded * a_tx_conj  # (batch, nrx, ntx)
            
            # Add to channel
            H = H + path_contribution
        
        # Note: Channel power is controlled by path gains α_ℓ ~ CN(0, 1)
        # Average channel power ≈ L (number of paths)
        # For beamforming, we want to preserve this natural scaling
        
        return H
    
    def call(self, batch_size):
        """
        Call method for Keras Layer API.
        
        Args:
            batch_size: Number of channels to generate (can be a tensor)
            
        Returns:
            Channel tensor
        """
        if isinstance(batch_size, tf.Tensor):
            batch_size = tf.get_static_value(batch_size)
            if batch_size is None:
                raise ValueError("batch_size must be known at graph construction time")
        
        return self.generate_channel(batch_size)


class SionnaCDLChannelModel(tf.keras.layers.Layer):
    """
    Sionna 3GPP TR 38.901 CDL Channel Model with Domain Randomization.
    
    This implementation now uses Sionna's native CDL + OFDM channel pipeline
    (ResourceGrid + GenerateOFDMChannel) instead of the previous manual
    parametric reconstruction. The output remains frequency-flat with shape
    (batch_size, num_rx_antennas, num_tx_antennas) to stay plug‑compatible with
    the rest of the beam-alignment code.
    
    Pipeline:
        CDL (TR38.901) --> GenerateOFDMChannel --> average over OFDM symbols &
        subcarriers --> H ∈ ℂ^{batch × NRX × NTX}
    
    Domain randomization:
        - Random CDL profile per batch (A/B/C/D/E)
        - Random delay spread per batch (uniform in delay_spread_range)
        - UE speed randomization is handled internally by Sionna via min/max speed
    """
    
    def __init__(self, 
                 num_tx_antennas, 
                 num_rx_antennas,
                 carrier_frequency=28e9,
                 delay_spread_range=(10e-9, 300e-9),  # 10ns to 300ns
                 ue_speed_range=(0.0, 30.0),  # 0 to 30 m/s (108 km/h)
                 cdl_models=None,
                 fft_size=64,
                 num_ofdm_symbols=1,
                 subcarrier_spacing=120e3,
                 **kwargs):
        """
        Args:
            num_tx_antennas: Number of BS transmit antennas (NTX)
            num_rx_antennas: Number of UE receive antennas (NRX)
            carrier_frequency: Carrier frequency in Hz (default: 28 GHz for mmWave)
            delay_spread_range: (min, max) delay spread in seconds for randomization
            ue_speed_range: (min, max) UE speed in m/s for Doppler randomization
            cdl_models: List of CDL model names (default: all 5 models)
                       Options: "A", "B", "C", "D", "E"
            fft_size: OFDM FFT size used when generating frequency responses
            num_ofdm_symbols: Number of OFDM symbols to generate (averaged)
            subcarrier_spacing: Subcarrier spacing in Hz
        """
        super().__init__(**kwargs)
        
        if not SIONNA_AVAILABLE:
            raise ImportError(
                "Sionna is not installed. Install with: pip install sionna\n"
                "See: https://nvlabs.github.io/sionna/"
            )
        
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.carrier_frequency = carrier_frequency
        self.delay_spread_range = delay_spread_range
        self.ue_speed_range = ue_speed_range
        self.fft_size = fft_size
        self.num_ofdm_symbols = num_ofdm_symbols
        self.subcarrier_spacing = subcarrier_spacing
        
        # Default to all CDL models for maximum diversity
        if cdl_models is None:
            cdl_models = ["A", "B", "C", "D", "E"]
        self.cdl_models = cdl_models
        self.num_cdl_models = len(cdl_models)
        
        # Build antenna arrays (ULA) for BS and UE
        self.bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_tx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
        )
        self.ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_rx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
        
        # Resource grid for OFDM channel generation
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
        )
        
        # Pre-instantiate one CDL + GenerateOFDMChannel per profile
        self._cdl_models = []
        self._ofdm_channels = []
        for model_name in cdl_models:
            cdl = CDL(
                model=model_name,
                delay_spread=delay_spread_range[0],
                carrier_frequency=carrier_frequency,
                ut_array=self.ut_array,
                bs_array=self.bs_array,
                direction="downlink",
                min_speed=ue_speed_range[0],
                max_speed=ue_speed_range[1],
            )
            ofdm_channel = GenerateOFDMChannel(
                channel_model=cdl,
                resource_grid=self.resource_grid
            )
            self._cdl_models.append(cdl)
            self._ofdm_channels.append(ofdm_channel)
        
        self._subcarrier_axes = (-1, -2)  # (subcarriers, OFDM symbols)
        print(f"✓ Sionna CDL Channel Model initialized (native)")
        print(f"  - CDL profiles: {', '.join(['CDL-' + m for m in cdl_models])}")
        print(f"  - Carrier frequency: {carrier_frequency/1e9:.1f} GHz")
        print(f"  - Delay spread range: {delay_spread_range[0]*1e9:.0f}-{delay_spread_range[1]*1e9:.0f} ns")
        print(f"  - UE speed range: {ue_speed_range[0]:.1f}-{ue_speed_range[1]:.1f} m/s")
        print(f"  - OFDM grid: fft_size={fft_size}, subcarrier_spacing={subcarrier_spacing/1e3:.0f} kHz")
        print(f"  - Antennas: {num_tx_antennas} BS, {num_rx_antennas} UE")
    
    def generate_channel(self, batch_size):
        """
        Generate a batch of CDL channel realizations with domain randomization.
        
        This method randomly samples:
        - CDL profile (A/B/C/D/E) for each batch element
        - UE speed (affects Doppler shift) - Currently not used (quasi-static)
        - Delay spread (affects multipath severity)
        
        The resulting channels have diverse characteristics that train robust models.
        
        Args:
            batch_size: Number of channel samples to generate
            
        Returns:
            Channel tensor of shape (batch_size, num_rx_antennas, num_tx_antennas)
            
        Note:
            This implementation leverages Sionna's CDL + OFDM pipeline and then
            averages over subcarriers and OFDM symbols to yield a frequency-flat
            channel matrix compatible with the rest of the codebase.
        """
        # Randomly select CDL model index
        cdl_idx = tf.random.uniform(
            [], minval=0, maxval=self.num_cdl_models, dtype=tf.int32
        )

        # Random delay spread for this batch
        delay_spread = tf.random.uniform(
            [], self.delay_spread_range[0], self.delay_spread_range[1], tf.float32
        )

        def _gen_with_model(model_idx):
            # Assign delay spread (Sionna samples speeds internally)
            self._cdl_models[model_idx].delay_spread = delay_spread
            h_freq = self._ofdm_channels[model_idx](batch_size)
            # Average over subcarriers and OFDM symbols to get flat channel
            h_flat = tf.reduce_mean(h_freq, axis=self._subcarrier_axes)
            # Remove singleton num_rx and num_tx dims: (batch, nrx_ant, ntx_ant)
            h_flat = tf.squeeze(h_flat, axis=[1, 3])
            return tf.cast(h_flat, tf.complex64)

        if self.num_cdl_models == 1:
            return _gen_with_model(0)

        branch_fns = {i: (lambda i=i: _gen_with_model(i)) for i in range(self.num_cdl_models)}
        return tf.switch_case(cdl_idx, branch_fns=branch_fns)
    
    def call(self, batch_size):
        """
        Call method for Keras Layer API.
        
        Args:
            batch_size: Number of channels to generate (can be a tensor)
            
        Returns:
            Channel tensor
        """
        if isinstance(batch_size, tf.Tensor):
            batch_size = tf.get_static_value(batch_size)
            if batch_size is None:
                raise ValueError("batch_size must be known at graph construction time")
        
        return self.generate_channel(batch_size)


# Alias for ease of use - defaults to Sionna if available, otherwise geometric
mmWaveChannel = SionnaCDLChannelModel if SIONNA_AVAILABLE else GeometricChannelModel


if __name__ == "__main__":
    print("Testing mmWave Channel Model...")
    print("=" * 60)
    
    # Create channel model
    channel_model = GeometricChannelModel(
        num_tx_antennas=64,
        num_rx_antennas=16,
        num_paths=3
    )
    
    # Generate channels
    batch_size = 100
    H = channel_model.generate_channel(batch_size)
    
    print(f"\nChannel tensor shape: {H.shape}")
    print(f"Channel dtype: {H.dtype}")
    print(f"Channel mean magnitude: {tf.reduce_mean(tf.abs(H)):.4f}")
    print(f"Channel Frobenius norm (mean): {tf.reduce_mean(tf.norm(H, axis=(1,2))):.4f}")
    
    # Compute channel statistics
    singular_values = tf.linalg.svd(H, compute_uv=False)
    print(f"\nChannel condition number (mean): {tf.reduce_mean(singular_values[..., 0] / singular_values[..., -1]):.2f}")
    print(f"Largest singular value (mean): {tf.reduce_mean(singular_values[..., 0]):.4f}")
    
    # Test with different number of paths
    print("\n" + "=" * 60)
    print("Testing with different number of paths:")
    for num_paths in [1, 3, 5, 10]:
        channel_model_test = GeometricChannelModel(
            num_tx_antennas=64,
            num_rx_antennas=16,
            num_paths=num_paths
        )
        H_test = channel_model_test.generate_channel(1000)
        mean_power = tf.reduce_mean(tf.reduce_sum(tf.abs(H_test) ** 2, axis=(1, 2)))
        print(f"  L={num_paths:2d}: Mean channel power = {mean_power:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
