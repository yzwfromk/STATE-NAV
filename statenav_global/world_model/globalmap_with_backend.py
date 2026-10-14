"""
Modified CMDbasedMap that uses SharedMemoryBackend

This shows how to modify CMDbasedMap to use the backend pattern.
The key insight: CMDbasedMap gets arrays from backend instead of creating them itself.
All methods work unchanged - they just use arrays from backend.

IMPORTANT: Call backend.cleanup() on shutdown to free shared memory!
"""

import math
import numpy as np
import atexit
try:
    import rospy
except ImportError:
    rospy = None

from statenav_global.world_model.globalmap import CMDbasedMap, BaseMap, cfg
from statenav_global.world_model.shared_memory_backend import SharedMemoryBackend, LocalMemoryBackend


class CMDbasedMapWithBackend(CMDbasedMap):
    """
    CMDbasedMap that uses SharedMemoryBackend
    
    Key changes:
    1. Accepts backend in __init__
    2. Gets arrays from backend instead of creating them
    3. Uses backend for synchronization
    4. All existing methods work unchanged!
    """
    
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y,
                 which_layer, preest_update_resolution, instab_limit,
                 load_Travformer=True,
                 backend=None, use_shared_memory=False):
        """
        Initialize CMDbasedMap with backend
        
        Args:
            backend: SharedMemoryBackend or LocalMemoryBackend instance
                     If None, creates appropriate backend based on use_shared_memory
            use_shared_memory: If True and backend=None, creates SharedMemoryBackend
        """
        # Initialize parent class (but don't create arrays yet)
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y,
                        which_layer, preest_update_resolution, instab_limit,
                        load_Travformer)
        
        # Set up backend
        if backend is None:
            if use_shared_memory:
                self.backend = SharedMemoryBackend(mode='writer')
            else:
                self.backend = LocalMemoryBackend()
        else:
            self.backend = backend
        
        self.use_shared_memory = use_shared_memory or isinstance(self.backend, SharedMemoryBackend)
        
        # Register cleanup handlers if using shared memory
        if self.use_shared_memory:
            atexit.register(self._cleanup_handler)
            if rospy is not None:
                rospy.on_shutdown(self._cleanup_handler)
        
        # Arrays will be set from backend in Initilize_map()
        # We'll override Initilize_map to use backend
    
    def _cleanup_handler(self):
        """Cleanup handler for atexit/ROS shutdown"""
        # Close the shared graph first (owned by the writer-side map); harmless
        # no-op when no graph was built yet (shutdown before Initilize_map).
        try:
            self._close_shared_graph()
        except Exception:
            pass  # Ignore errors during shutdown
        if hasattr(self, 'backend') and self.use_shared_memory:
            try:
                self.backend.cleanup()
            except Exception:
                pass  # Ignore errors during shutdown

    def Initilize_map(self):
        """
        Initialize maps using backend
        
        This replaces the original Initilize_map to use backend arrays.
        Note: If using shared memory, arrays are already initialized in robo_centric_map_callback.
        """
        if self.is_RobocentricMap_received:
            if not self.is_ElevationMap_built:
                self.is_ElevationMap_built = True
                if hasattr(self, 'backend'):
                    self.backend.set_elev_map_built(True)
            
            # Calculate dimensions
            self.map_size_rows = int(self.map_size_x / self.map_resolution)
            self.map_size_cols = int(self.map_size_y / self.map_resolution)
            
            # Initialize backend with dimensions
            self.backend.initialize_arrays(
                elev_map_rows=self.map_size_rows,
                elev_map_cols=self.map_size_cols,
                trav_map_rows=self.map_size_rows,
                trav_map_cols=self.map_size_cols,
                trav_map_layers=self.TraversabilityMap_size_layers,
                map_resolution=self.map_resolution,
                elev_map_xmin=self.map_xmin,
                elev_map_xmax=self.map_xmax,
                elev_map_ymin=self.map_ymin,
                elev_map_ymax=self.map_ymax
            )
            
            # Get arrays from backend (these point to shared memory if using it).
            # Elevation only: traversability lives in the shared graph built below.
            self.ElevationMap = self.backend.get_elevation_map()
            self.ElevMap_MetaInfo = self.backend.get_elev_meta_info()

            # The initial fill IS map content, so the version leaves initialization at
            # >= 1: a reader can then tell "map written" from the 0 of a fresh block.
            self.bump_map_version()

            # Build the lattice-aligned shared graph now that coarse-lattice
            # geometry is known (writer side, every planner type). Its creation seeds
            # the neutral interior and the non-inferable too-risky border.
            # CMDbasedMap._build_shared_graph owns all graph logic and imports; this
            # is only the construction hook.
            self._build_shared_graph()

    
    def Update_map(self, path_plan=None, visualize_map=False):
        """
        Update map with synchronization
        
        This wraps the original Update_map with locking
        """
        if self.use_shared_memory:
            # Acquire write lock before updating
            self.backend.acquire_write_lock()
        
        try:
            # Call parent's Update_map (all existing code works!)
            super().Update_map(path_plan, visualize_map)
            
            # After updates, increment version. Not gated on use_shared_memory:
            # LocalMemoryBackend carries the same counter, and every other map-write
            # path bumps it unconditionally too (BaseMap.bump_map_version).
            self.bump_map_version()
        
        finally:
            if self.use_shared_memory:
                # Always release lock
                self.backend.release_write_lock()
    
    def get_backend(self):
        """Get backend instance (for access to metadata, version, etc.)"""
        return self.backend
    
    def cleanup(self):
        """Explicit cleanup (also called automatically on shutdown)"""
        # Close the shared graph (no-op when none was built), then the backend.
        self._close_shared_graph()
        if self.use_shared_memory:
            self.backend.cleanup()


class CMDbasedMapReaderProxy(CMDbasedMap):
    """
    Proxy class for Planner to access CMDbasedMap via shared memory
    
    This inherits from CMDbasedMap so all methods work unchanged.
    The only difference: arrays come from shared memory backend.
    """
    
    def __init__(self, backend: SharedMemoryBackend):
        """
        Initialize proxy with shared memory backend
        
        Args:
            backend: SharedMemoryBackend instance (mode='reader')
        """
        self.backend = backend
        
        # Wait for writer to initialize metadata and shared memory
        # Then initialize backend arrays
        self._wait_and_initialize_backend()
        
        # Get dimensions and bounds from backend (now initialized)
        dims = backend.get_map_dimensions()
        bounds = backend.get_map_bounds()
        
        # Initialize parent CMDbasedMap with minimal parameters
        # (We'll override the arrays anyway)
        super().__init__(
            env_xmin=bounds[0],
            env_xmax=bounds[1],
            env_ymin=bounds[2],
            env_ymax=bounds[3],
            goal_x=0.0,  # Will be set separately if needed
            goal_y=0.0,
            which_layer=[],  # Not used in reader
            preest_update_resolution=cfg.trav_estimation_resoultion,  # Must match writer
            instab_limit=0.0,  # Not used in reader
            load_Travformer=False,  # Don't load models in reader
        )
        
        # Register cleanup handlers
        atexit.register(self._cleanup_handler)
        if rospy is not None:
            rospy.on_shutdown(self._cleanup_handler)
        
        # Override arrays with shared memory arrays (elevation only; traversability is
        # read from the shared graph through graph_cmd_at / the planner's own reader).
        self.ElevationMap = backend.get_elevation_map()
        self.ElevMap_MetaInfo = backend.get_elev_meta_info()

        # Set map parameters from backend metadata
        self.map_resolution = backend.get_map_resolution()
        bounds = backend.get_map_bounds()
        self.map_xmin = bounds[0]
        self.map_xmax = bounds[1]
        self.map_ymin = bounds[2]
        self.map_ymax = bounds[3]
        
        # Set dimensions
        dims = backend.get_map_dimensions()
        self.TraversabilityMap_size_layers = dims['trav_map_layers']
        self.map_size_rows = dims['elev_map_rows']
        self.map_size_cols = dims['elev_map_cols']
        
        # Size attributes for consistency (calculated from bounds)
        self.map_size_x = self.map_xmax - self.map_xmin
        self.map_size_y = self.map_ymax - self.map_ymin

        # Map parameters (same as CMDbasedMap)
        self.TraversabilityMap_theta_min = -np.pi
        self.TraversabilityMap_theta_resolution = np.pi / 4
        self.TraversabilityMap_theta_max = np.pi - self.TraversabilityMap_theta_resolution

        # is_TraversabilityMap_built / is_ElevationMap_built
        # are defined as properties below — they always read live from the backend.
        # No snapshot assignments here; the parent's __init__ writes are absorbed by no-op setters.
        self.is_ElevMap_MetaInfo_built = True  # MetaInfo arrays are available from shared memory
    
    # ── Live-read properties: always reflect current backend state ───────────
    # The parent's __init__ writes False to these; the no-op setters absorb those
    # writes so instance attributes never shadow the backend values.

    @property
    def is_TraversabilityMap_built(self):
        return self.backend.get_trav_map_built()

    @is_TraversabilityMap_built.setter
    def is_TraversabilityMap_built(self, value):
        pass  # reader never owns this flag; writer updates shared memory directly

    @property
    def is_ElevationMap_built(self):
        return self.backend.get_elev_map_built()

    @is_ElevationMap_built.setter
    def is_ElevationMap_built(self, value):
        pass  # same as above

    
    def _wait_and_initialize_backend(self, timeout: float = 120.0, check_interval: float = 0.1):
        """
        Wait for writer to initialize metadata AND create shared memory blocks, then initialize backend arrays.
        
        Args:
            timeout: Maximum time to wait for writer initialization (seconds)
            check_interval: Time between checks (seconds)
        """
        import time
        from multiprocessing import shared_memory
        
        start_time = time.time()
        if rospy is not None:
            rospy.loginfo("[PathPlanner] Waiting for WorldModel to initialize shared memory...")
        
        while time.time() - start_time < timeout:
            # Check if metadata has been initialized (dimensions > 0)
            dims = self.backend.get_map_dimensions()
            if dims['trav_map_rows'] > 0 and dims['elev_map_rows'] > 0:
                # Metadata is ready, check if shared memory blocks exist
                # Try to attach to all required shared memory blocks to verify they exist
                
                # TODO: Why?
                try:
                    test_shm_elev = shared_memory.SharedMemory(name='elev_map_shm')
                    test_shm_elev_meta = shared_memory.SharedMemory(name='elev_meta_shm')
                    # Close test connections
                    test_shm_elev.close()
                    test_shm_elev_meta.close()
                    
                    # All shared memory blocks exist, now initialize arrays (this will attach properly)
                    bounds = self.backend.get_map_bounds()
                    self.backend.initialize_arrays(
                        elev_map_rows=dims['elev_map_rows'],
                        elev_map_cols=dims['elev_map_cols'],
                        trav_map_rows=dims['trav_map_rows'],
                        trav_map_cols=dims['trav_map_cols'],
                        trav_map_layers=dims['trav_map_layers'],
                        map_resolution=self.backend.get_map_resolution(),
                        elev_map_xmin=bounds[0],
                        elev_map_xmax=bounds[1],
                        elev_map_ymin=bounds[2],
                        elev_map_ymax=bounds[3]
                    )
                    if rospy is not None:
                        rospy.loginfo("[Reader] Successfully attached to shared memory")
                    return
                except FileNotFoundError:
                    # Shared memory blocks don't exist yet, keep waiting
                    # This can happen if metadata is set before blocks are created
                    pass
                except Exception as e:
                    # Other errors - log and continue waiting
                    if rospy is not None:
                        rospy.logwarn_throttle(5.0, f"[Reader] Error checking shared memory: {e}")
            
            time.sleep(check_interval)
            if rospy is not None:
                rospy.loginfo_throttle(2.0, "[Reader] Still waiting for WorldModel to initialize shared memory...")
        
        raise RuntimeError(f"Timeout waiting for WorldModel to initialize shared memory ({timeout}s)")
    
    def check_version(self):
        """Check if maps have been updated"""
        return self.backend.get_version()
    
    def acquire_read_lock(self):
        """
        Acquire read lock (OPTIONAL - for consistency guarantees)
        
        Read lock is SHARED - multiple readers can hold it.
        
        Use when:
        - Reading multiple elements that must be consistent together
        - Want to prevent reading during bulk updates
        
        Don't need for:
        - Individual element reads (atomic): trav_map[row, col, layer, 0]
        - Most planning operations (they read individual elements)
        """
        self.backend.acquire_read_lock()
    
    def release_read_lock(self):
        """Release read lock"""
        self.backend.release_read_lock()
    
    def _cleanup_handler(self):
        """Cleanup handler for atexit/ROS shutdown"""
        # Close graph_cmd_at's reader handles (a reader never owns or unlinks the graph).
        try:
            self._close_shared_graph()
        except Exception:
            pass  # Ignore errors during shutdown
        if hasattr(self, 'backend'):
            try:
                self.backend.cleanup()
            except Exception:
                pass  # Ignore errors during shutdown

    def cleanup(self):
        """Explicit cleanup (also called automatically on shutdown)"""
        self._close_shared_graph()
        self.backend.cleanup()

