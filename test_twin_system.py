#!/usr/bin/env python3
"""Comprehensive testing of Twin Model System for bugs and design issues"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import traceback
from copy import deepcopy
import sys

# Import twin system
from twin_system import TwinModelSystem
from data_fetcher import BinanceDataFetcher
from portfolio_env import PortfolioTradingEnvironment

print("="*60)
print("TWIN MODEL SYSTEM - BUG ANALYSIS & TESTING")
print("="*60)

# Track issues found
issues_found = []

# TEST 1: Check for partial implementations
print("\n1. CHECKING FOR PARTIAL IMPLEMENTATIONS")
print("-" * 40)

def check_method_implementation(obj, method_name):
    """Check if method exists and is implemented"""
    if not hasattr(obj, method_name):
        return f"Missing method: {method_name}"
    
    method = getattr(obj, method_name)
    if method is None:
        return f"Method {method_name} is None"
    
    # Check if it's a stub (contains pass or NotImplementedError)
    import inspect
    try:
        source = inspect.getsource(method)
        if 'NotImplementedError' in source:
            return f"Method {method_name} raises NotImplementedError"
        if source.strip().endswith('pass'):
            return f"Method {method_name} is just 'pass'"
    except:
        pass
    
    return None

# Check TwinModelSystem
twin = TwinModelSystem()
required_methods = [
    'initialize', 'predict', 'update', '_should_retrain', '_should_hyperopt',
    '_should_switch', '_evaluate_and_switch', '_start_background_training',
    '_background_training', '_start_background_hyperopt', '_background_hyperopt',
    '_train_model', '_test_model', '_hyperopt', '_prepare_observation',
    'get_status', 'save_state', 'load_state'
]

for method in required_methods:
    issue = check_method_implementation(twin, method)
    if issue:
        print(f"❌ {issue}")
        issues_found.append(issue)
    else:
        print(f"✅ {method} implemented")

# TEST 2: Thread Safety Analysis
print("\n2. THREAD SAFETY ANALYSIS")
print("-" * 40)

# Check for race conditions
print("Checking for proper locking mechanisms...")

import inspect
source = inspect.getsource(TwinModelSystem)

# Check lock usage
if 'self.lock' in source:
    print("✅ Lock object found")
    
    # Check if lock is used in critical sections
    critical_methods = ['predict', '_evaluate_and_switch', '_background_training', '_background_hyperopt']
    for method in critical_methods:
        method_source = inspect.getsource(getattr(TwinModelSystem, method))
        if 'with self.lock:' in method_source or 'self.lock.acquire()' in method_source:
            print(f"✅ {method} uses lock")
        else:
            print(f"⚠️ {method} may have race condition")
            issues_found.append(f"{method} doesn't use lock properly")
else:
    print("❌ No lock mechanism found")
    issues_found.append("No threading lock found")

# TEST 3: Model Cloning Verification
print("\n3. MODEL CLONING VERIFICATION")
print("-" * 40)

try:
    # Create dummy data
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='1h'),
        'open': np.random.randn(100) * 100 + 50000,
        'high': np.random.randn(100) * 100 + 50100,
        'low': np.random.randn(100) * 100 + 49900,
        'close': np.random.randn(100) * 100 + 50000,
        'volume': np.random.randn(100) * 1000000 + 5000000
    })
    
    twin_test = TwinModelSystem()
    
    # Check initialization
    print("Testing model initialization...")
    twin_test.initialize(df)
    
    # Verify models are identical initially
    if twin_test.model_a is twin_test.model_b:
        print("❌ Model B is same object as Model A (not cloned)")
        issues_found.append("Models not properly cloned")
    else:
        print("✅ Model B is separate object")
        
        # Check if they have same parameters initially
        if hasattr(twin_test.model_a, 'policy'):
            params_a = twin_test.model_a.policy.state_dict()
            params_b = twin_test.model_b.policy.state_dict()
            
            # Check if parameters are equal but not same object
            same_values = all(torch.equal(params_a[key], params_b[key]) 
                             for key in params_a.keys())
            same_objects = any(params_a[key] is params_b[key] 
                              for key in params_a.keys())
            
            if same_values and not same_objects:
                print("✅ Models properly cloned (same values, different objects)")
            else:
                print("⚠️ Model cloning may have issues")
        
except Exception as e:
    print(f"❌ Error in model cloning test: {e}")
    issues_found.append(f"Model cloning error: {e}")

# TEST 4: Performance Tracking
print("\n4. PERFORMANCE TRACKING VERIFICATION")
print("-" * 40)

# Check deque implementation
if hasattr(twin, 'model_a_returns') and hasattr(twin, 'model_b_returns'):
    print("✅ Performance tracking deques exist")
    
    # Check maxlen
    from collections import deque
    if isinstance(twin.model_a_returns, deque):
        if twin.model_a_returns.maxlen == twin.performance_window:
            print(f"✅ Deque maxlen set correctly ({twin.performance_window})")
        else:
            print(f"❌ Deque maxlen mismatch")
            issues_found.append("Performance deque maxlen incorrect")
else:
    print("❌ Performance tracking not properly initialized")
    issues_found.append("Missing performance tracking deques")

# TEST 5: Observation Preparation
print("\n5. OBSERVATION PREPARATION CHECK")
print("-" * 40)

# Check _prepare_observation
obs_method = twin._prepare_observation
obs_source = inspect.getsource(obs_method)

if 'random' in obs_source or 'randn' in obs_source:
    print("⚠️ WARNING: _prepare_observation uses random data (placeholder)")
    issues_found.append("_prepare_observation is placeholder implementation")
else:
    print("✅ _prepare_observation appears implemented")

# TEST 6: Data Buffer Management
print("\n6. DATA BUFFER MANAGEMENT")
print("-" * 40)

if hasattr(twin, 'data_buffer'):
    print(f"✅ Data buffer exists (maxlen={twin.data_buffer.maxlen})")
    
    # Test adding data
    try:
        for i in range(600):  # More than maxlen
            twin.data_buffer.append({'test': i})
        
        if len(twin.data_buffer) == 500:
            print("✅ Data buffer correctly limits to 500 entries")
        else:
            print(f"❌ Data buffer has {len(twin.data_buffer)} entries (expected 500)")
            issues_found.append("Data buffer size management issue")
    except Exception as e:
        print(f"❌ Error testing data buffer: {e}")
        issues_found.append(f"Data buffer error: {e}")
else:
    print("❌ No data buffer found")
    issues_found.append("Missing data buffer")

# TEST 7: Timing Logic
print("\n7. TIMING LOGIC VERIFICATION")
print("-" * 40)

# Test interval calculations
twin.last_retrain = datetime.now() - timedelta(hours=25)
twin.last_hyperopt = datetime.now() - timedelta(hours=170)
twin.last_switch_check = datetime.now() - timedelta(hours=7)

now = datetime.now()

should_retrain = twin._should_retrain(now)
should_hyperopt = twin._should_hyperopt(now)
should_switch = twin._should_switch(now)

print(f"Should retrain (>24h): {should_retrain} (Expected: True)")
print(f"Should hyperopt (>168h): {should_hyperopt} (Expected: True)")
print(f"Should switch (>6h): {should_switch} (Expected: True)")

if not should_retrain or not should_hyperopt or not should_switch:
    issues_found.append("Timing logic error")

# TEST 8: Model Switching Logic
print("\n8. MODEL SWITCHING LOGIC")
print("-" * 40)

# Check switching threshold
switch_source = inspect.getsource(twin._evaluate_and_switch)
if '0.5' in switch_source or '0.005' in switch_source:
    print("✅ Switching threshold found (0.5%)")
else:
    print("⚠️ Switching threshold not clearly defined")
    issues_found.append("Switching threshold unclear")

# TEST 9: Background Training Safety
print("\n9. BACKGROUND TRAINING SAFETY")
print("-" * 40)

# Check if is_training flag is used
if hasattr(twin, 'is_training'):
    print("✅ Training flag exists")
    
    # Check if it prevents concurrent training
    bg_training_source = inspect.getsource(twin._start_background_training)
    if 'if not self.is_training:' in inspect.getsource(twin.update):
        print("✅ Prevents concurrent training")
    else:
        print("⚠️ May allow concurrent training")
        issues_found.append("Concurrent training not prevented")
else:
    print("❌ No training flag")
    issues_found.append("Missing training flag")

# TEST 10: Memory Leaks
print("\n10. MEMORY LEAK CHECK")
print("-" * 40)

# Check for proper cleanup
if hasattr(twin, '__del__'):
    print("✅ Destructor defined")
else:
    print("⚠️ No destructor - may have memory leaks")

# Check thread daemon status
if hasattr(twin, 'training_thread'):
    print("⚠️ Thread management needs verification")

# DESIGN CONCERNS
print("\n" + "="*60)
print("DESIGN CONCERNS & RECOMMENDATIONS")
print("="*60)

design_concerns = [
    {
        'issue': 'Placeholder Observation',
        'concern': '_prepare_observation returns random data',
        'recommendation': 'Implement proper feature extraction from market state'
    },
    {
        'issue': 'No Model Versioning',
        'concern': 'No tracking of which model version is active',
        'recommendation': 'Add model versioning and logging'
    },
    {
        'issue': 'Fixed Intervals',
        'concern': 'Hard-coded retraining intervals may not be optimal',
        'recommendation': 'Make intervals configurable or adaptive'
    },
    {
        'issue': 'Limited Performance Metrics',
        'concern': 'Only tracks returns, not risk metrics',
        'recommendation': 'Add Sharpe ratio, drawdown tracking'
    },
    {
        'issue': 'No Failsafe',
        'concern': 'No fallback if both models fail',
        'recommendation': 'Add emergency stop or baseline model'
    },
    {
        'issue': 'Thread Cleanup',
        'concern': 'Background threads may not clean up properly',
        'recommendation': 'Add proper thread lifecycle management'
    }
]

for concern in design_concerns:
    print(f"\n⚠️ {concern['issue']}")
    print(f"   Concern: {concern['concern']}")
    print(f"   Fix: {concern['recommendation']}")

# SUMMARY
print("\n" + "="*60)
print("TESTING SUMMARY")
print("="*60)

if not issues_found:
    print("✅ No critical bugs found!")
else:
    print(f"❌ Found {len(issues_found)} issues:")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")

print("\n📊 Component Status:")
print(f"  Core Methods: {'✅' if not any('Missing method' in i for i in issues_found) else '❌'}")
print(f"  Thread Safety: {'✅' if not any('lock' in i.lower() for i in issues_found) else '⚠️'}")
print(f"  Model Cloning: {'✅' if not any('clon' in i.lower() for i in issues_found) else '❌'}")
print(f"  Data Management: {'✅' if not any('buffer' in i.lower() for i in issues_found) else '❌'}")
print(f"  Timing Logic: {'✅' if not any('timing' in i.lower() for i in issues_found) else '❌'}")

print("\n" + "="*60)