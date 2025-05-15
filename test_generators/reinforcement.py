"""
Reinforcement Learning-Based Test Generation Module

This module creates test cases using reinforcement learning principles
to intelligently explore the state space of the system under test.
"""
import re
import random
from typing import List, Dict, Any, Set, Tuple
import nltk

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def generate_reinforcement_tests(source_type: str, source_content: str) -> List[Dict[str, Any]]:
    """
    Generate tests using reinforcement learning techniques from source code or requirements.
    
    Args:
        source_type: 'code' or 'requirements'
        source_content: The source code or requirements text
        
    Returns:
        List of test cases with name, description, and steps
    """
    if source_type == 'code':
        return _generate_from_code(source_content)
    else:  # source_type == 'requirements'
        return _generate_from_requirements(source_content)

def _extract_states_and_actions(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract potential states and actions from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple containing lists of states and actions
    """
    # Extract potential states
    state_patterns = [
        r'(?:page|screen|view|form|state)\s+(?:for|called|named)?\s*[\'"]?([a-zA-Z0-9_\s]+)[\'"]?',
        r'(?:the|a|an)\s+([a-zA-Z0-9_\s]+)\s+(?:page|screen|view|form|state)'
    ]
    
    states = set()
    for pattern in state_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            state_name = match.strip()
            if state_name and len(state_name) > 2 and state_name not in ['the', 'and', 'or']:
                states.add(state_name)
    
    # If no states found, infer from common application patterns
    if not states:
        potential_states = [
            "login", "register", "dashboard", "home", "profile", "settings",
            "search results", "details", "edit", "create", "list", "summary"
        ]
        
        for state in potential_states:
            if state in text.lower():
                states.add(state)
    
    # Ensure we have at least some states to work with
    if not states:
        states = {"initial state", "main state", "final state"}
    
    # Extract potential actions
    action_patterns = [
        r'(?:user|system|customer|client|actor)\s+(?:can|should|must|may|will)\s+([a-z]+)',
        r'(?:can|should|must|may|will)\s+([a-z]+)\s+(?:the|a|an)',
        r'ability\s+to\s+([a-z]+)',
        r'(?:clicking|pressing|selecting)\s+(?:the|a|an)\s+([a-z]+)',
        r'(?:button|link|menu)\s+to\s+([a-z]+)'
    ]
    
    actions = set()
    for pattern in action_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            action_name = match.strip()
            if action_name and len(action_name) > 2 and action_name not in ['be', 'have', 'do', 'get', 'make', 'see', 'use', 'the', 'and', 'can', 'will']:
                actions.add(action_name)
    
    # If no actions found, infer from common patterns
    if not actions:
        common_actions = ["login", "logout", "submit", "search", "create", "edit", "delete", "view", 
                         "browse", "select", "navigate", "filter", "sort", "upload", "download"]
        
        for action in common_actions:
            if action in text.lower():
                actions.add(action)
    
    # Ensure we have at least some actions to work with
    if not actions:
        actions = {"click", "navigate", "input", "submit", "verify"}
    
    return list(states), list(actions)

def _generate_from_requirements(requirements: str) -> List[Dict[str, Any]]:
    """
    Generate reinforcement learning-based tests from requirements text.
    
    Args:
        requirements: Requirements text
        
    Returns:
        List of test cases
    """
    test_cases = []
    
    # Extract states and actions from requirements
    states, actions = _extract_states_and_actions(requirements)
    
    # Generate state-action exploration test
    state_exploration_steps = [
        "Initialize the system to its starting state",
        "Identify all possible actions from the current state",
        "Select the action that leads to the least explored state",
        "Execute the selected action and observe the resulting state",
        "Update exploration statistics for the state-action pair",
        "Repeat steps 2-5 until all state-action pairs are explored or a termination condition is met"
    ]
    
    test_cases.append({
        'name': "State-space exploration test",
        'description': "Systematically explore the application's state space using reinforcement learning principles",
        'preconditions': "Application is in a known initial state",
        'expected_result': "All reachable states are discovered and all valid actions are identified",
        'steps': state_exploration_steps
    })
    
    # Generate boundary condition discovery test
    boundary_steps = [
        "Initialize the system to its starting state",
        "Identify boundary conditions in the requirements",
        "Prioritize actions that target these boundary conditions",
        "Execute actions that approach boundary values",
        "Progressively refine actions to pinpoint exact boundary conditions",
        "Verify system behavior at boundary conditions"
    ]
    
    test_cases.append({
        'name': "Boundary condition discovery test",
        'description': "Discover and test boundary conditions using adaptive learning",
        'preconditions': "Application is in a known initial state",
        'expected_result': "All boundary conditions are identified and tested",
        'steps': boundary_steps
    })
    
    # Generate optimal path discovery test
    optimal_path_steps = [
        "Define the goal state to reach",
        "Initialize the system to its starting state",
        "Use Q-learning to explore paths toward the goal",
        "Progressively favor actions with higher expected rewards",
        "Identify the optimal path to the goal state",
        "Verify the optimal path functions correctly"
    ]
    
    # Create a specific test case for each identified state as a potential goal
    for i, state in enumerate(states[:3]):  # Limit to 3 states
        target_state = state
        state_specific_steps = [step.replace("the goal state", f"the '{target_state}' state") 
                                for step in optimal_path_steps]
        
        test_cases.append({
            'name': f"Optimal path to {target_state} test",
            'description': f"Discover and verify the optimal path to reach the {target_state} state",
            'preconditions': "Application is in a known initial state",
            'expected_result': f"The optimal path to the {target_state} state is identified and verified",
            'steps': state_specific_steps
        })
    
    # Generate negative scenario discovery test
    negative_steps = [
        "Identify critical actions in the system",
        "For each action, generate scenarios where the action should fail",
        "Execute the action in these scenarios with varying inputs",
        "Learn from the system responses to identify edge cases",
        "Focus exploration on areas with unexpected behavior",
        "Verify that all negative scenarios are handled appropriately"
    ]
    
    test_cases.append({
        'name': "Negative scenario discovery test",
        'description': "Intelligently discover and test negative scenarios and edge cases",
        'preconditions': "Application is available for testing",
        'expected_result': "All negative scenarios are identified and handle errors appropriately",
        'steps': negative_steps
    })
    
    # Generate input sensitivity test
    sensitivity_steps = [
        "Identify input fields in the system",
        "Apply reinforcement learning to discover input sensitivity patterns",
        "Generate diverse inputs focusing on boundary values",
        "Progressively refine inputs based on system responses",
        "Identify patterns of input sensitivity",
        "Verify that all inputs are properly validated"
    ]
    
    test_cases.append({
        'name': "Input sensitivity discovery test",
        'description': "Discover and test input sensitivity patterns using reinforcement learning",
        'preconditions': "Application is available for testing",
        'expected_result': "Input sensitivity patterns are identified and all inputs are properly validated",
        'steps': sensitivity_steps
    })
    
    return test_cases

def _generate_from_code(code: str) -> List[Dict[str, Any]]:
    """
    Generate reinforcement learning-based tests from source code.
    
    Args:
        code: Source code
        
    Returns:
        List of test cases
    """
    test_cases = []
    
    # Extract functions and parameters
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
    functions = re.findall(function_pattern, code)
    
    # If no functions found, create generic test cases
    if not functions:
        return _generate_generic_tests()
    
    # Filter out private functions
    public_functions = [(name, params) for name, params in functions if not name.startswith('_')]
    
    # If no public functions, create generic test cases
    if not public_functions:
        return _generate_generic_tests()
    
    # Generate parameter exploration test
    for i, (func_name, params) in enumerate(public_functions[:3]):  # Limit to 3 functions
        param_list = [p.strip() for p in params.split(',') if p.strip()]
        
        # Process parameters to remove type hints and default values
        clean_params = []
        for p in param_list:
            # Remove type hints
            if ':' in p:
                p = p.split(':', 1)[0].strip()
            # Remove default values
            if '=' in p:
                p = p.split('=', 1)[0].strip()
            clean_params.append(p)
        
        if clean_params:
            param_exploration_steps = [
                f"Identify input parameter space for function {func_name}",
                "Start with simple values for all parameters",
                "Use reinforcement learning to explore parameter combinations",
                "Adjust parameter values based on function behavior",
                "Prioritize exploration of boundary cases",
                "Identify optimal and edge-case parameter values",
                "Verify function behavior with these identified values"
            ]
            
            test_cases.append({
                'name': f"Parameter space exploration for {func_name}",
                'description': f"Systematically explore the parameter space of the {func_name} function using reinforcement learning",
                'preconditions': f"Function {func_name} is accessible",
                'expected_result': f"Complete parameter space is explored and function behavior is verified across all input ranges",
                'steps': param_exploration_steps
            })
    
    # Generate functional sequence exploration test
    if len(public_functions) > 1:
        sequence_exploration_steps = [
            "Identify all public functions in the codebase",
            "Start with a known initial state",
            "Use reinforcement learning to explore function call sequences",
            "Learn which sequences lead to stable states and which lead to errors",
            "Prioritize exploration of less-visited sequences",
            "Identify valid function call sequences and invalid ones",
            "Verify system behavior with both valid and invalid sequences"
        ]
        
        test_cases.append({
            'name': "Function sequence exploration test",
            'description': "Discover valid and invalid function call sequences using reinforcement learning",
            'preconditions': "System is in a known initial state",
            'expected_result': "All valid function sequences are identified and invalid sequences are properly handled",
            'steps': sequence_exploration_steps
        })
    
    # Add general reinforcement learning test cases
    generic_cases = _generate_generic_tests()
    test_cases.extend(generic_cases)
    
    return test_cases

def _generate_generic_tests() -> List[Dict[str, Any]]:
    """
    Generate generic reinforcement learning-based tests.
    
    Returns:
        List of test cases
    """
    test_cases = []
    
    # Generate exploration vs. exploitation test
    exploration_steps = [
        "Define the testing goal and reward function",
        "Initialize the system to its starting state",
        "Start with a high exploration rate (random actions)",
        "Gradually decrease exploration and increase exploitation of learned paths",
        "Focus on actions with highest expected rewards",
        "Verify key paths and edge cases discovered during exploration"
    ]
    
    test_cases.append({
        'name': "Exploration vs. exploitation balance test",
        'description': "Balance between exploring new states and exploiting known paths using reinforcement learning",
        'preconditions': "System is in a known initial state",
        'expected_result': "Optimal testing coverage is achieved balancing breadth and depth",
        'steps': exploration_steps
    })
    
    # Generate adaptive path testing
    adaptive_steps = [
        "Define multiple goals to achieve in the system",
        "Start with random exploration to discover the state space",
        "Learn which actions lead toward each goal",
        "Develop multiple paths to each goal",
        "Identify the most efficient paths to each goal",
        "Verify all discovered paths for correctness"
    ]
    
    test_cases.append({
        'name': "Adaptive path testing",
        'description': "Adaptively discover and test multiple paths to system goals",
        'preconditions': "System is in a known initial state",
        'expected_result': "Multiple valid paths to each goal are discovered and verified",
        'steps': adaptive_steps
    })
    
    # Generate anomaly detection test
    anomaly_steps = [
        "Establish a baseline of normal system behavior",
        "Use reinforcement learning to explore unusual action sequences",
        "Identify actions that produce unexpected results",
        "Focus testing on areas with divergent behavior",
        "Classify anomalies as bugs or intentional edge cases",
        "Verify proper handling of all identified edge cases"
    ]
    
    test_cases.append({
        'name': "Anomaly detection test",
        'description': "Use reinforcement learning to identify and test anomalous system behavior",
        'preconditions': "System is operational with known baseline behavior",
        'expected_result': "All behavioral anomalies are identified and properly classified",
        'steps': anomaly_steps
    })
    
    return test_cases