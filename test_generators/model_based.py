"""
Model-Based Testing Implementation

This module implements model-based test generation for state-based systems,
creating tests that verify state transitions and system behaviors by exploring
possible states, actions, and transitions.
"""
import re
import nltk
import networkx as nx
from typing import List, Dict, Any, Tuple

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def generate_tests(input_type: str, input_content: str) -> List[Dict[str, Any]]:
    """
    Generate model-based tests from source code or requirements.
    
    Args:
        input_type: 'code' or 'requirements'
        input_content: The source code or requirements text
        
    Returns:
        List of test cases with name, description, steps, etc.
    """
    # Extract states, actions, and transitions
    states, actions, transitions = _extract_model_components(input_type, input_content)
    
    # Create a directed graph representing the state model
    state_model = _build_state_model(states, transitions)
    
    # Generate test cases from the model
    test_cases = _generate_test_cases_from_model(state_model, actions, transitions)
    
    return test_cases

def _extract_model_components(input_type: str, input_content: str) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    """
    Extract states, actions, and transitions from requirements or code.
    
    Args:
        input_type: 'code' or 'requirements'
        input_content: The source code or requirements text
        
    Returns:
        Tuple containing lists of states, actions, and transitions
    """
    states = []
    actions = []
    transitions = []
    
    if input_type == 'requirements':
        # Extract from requirements text
        sentences = nltk.sent_tokenize(input_content)
        
        # Extract potential states
        state_patterns = [
            r'(?:state|screen|page|view)\s+(?:called|named)?\s*["\']?([a-zA-Z0-9_\s]+)["\']?',
            r'(?:in|at|on)\s+(?:the)?\s*([a-zA-Z0-9_\s]+)\s+(?:state|screen|page|view)',
            r'(?:system|user)\s+(?:is|should be)\s+(?:in|at)\s+(?:the)?\s*([a-zA-Z0-9_\s]+)',
            r'(?:when|if)\s+(?:the)?\s*(?:system|user)\s+(?:is|in)\s+(?:the)?\s*([a-zA-Z0-9_\s]+)\s+(?:state|screen|page|view)'
        ]
        
        for sentence in sentences:
            for pattern in state_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    state = match.strip()
                    if state and state not in states and len(state) > 2 and not any(x in state.lower() for x in ['the', 'a', 'an']):
                        states.append(state)
        
        # Extract potential actions
        action_patterns = [
            r'(?:user|system)\s+(?:can|should|must|will)\s+([a-z]+(?:\s+[a-z]+){0,5})',
            r'(?:clicking|pressing|selecting|entering|submitting|uploading)\s+([a-z]+(?:\s+[a-z]+){0,3})',
            r'(?:by|when|after)\s+([a-z]+ing\s+[a-z]+(?:\s+[a-z]+){0,3})'
        ]
        
        for sentence in sentences:
            for pattern in action_patterns:
                matches = re.findall(pattern, sentence.lower(), re.IGNORECASE)
                for match in matches:
                    action = match.strip()
                    if action and action not in actions and len(action) > 2:
                        actions.append(action)
        
        # Extract potential transitions
        for i, sentence in enumerate(sentences):
            for action in actions:
                if action in sentence.lower():
                    # Look for state mentions
                    source_state = None
                    target_state = None
                    
                    for state in states:
                        if state.lower() in sentence.lower():
                            if not source_state:
                                source_state = state
                            elif not target_state:
                                target_state = state
                    
                    if source_state and not target_state:
                        # If only one state mentioned, try to find another in adjacent sentences
                        if i > 0:
                            for state in states:
                                if state.lower() in sentences[i-1].lower() and state != source_state:
                                    target_state = state
                                    break
                                    
                        if not target_state and i < len(sentences) - 1:
                            for state in states:
                                if state.lower() in sentences[i+1].lower() and state != source_state:
                                    target_state = state
                                    break
                    
                    # If we found or can infer both states, add a transition
                    if source_state and target_state:
                        transitions.append({
                            'source': source_state,
                            'target': target_state,
                            'action': action,
                            'guard': '',
                            'expected': f"System transitions from {source_state} to {target_state}"
                        })
    
    elif input_type == 'code':
        # Extract from source code (simplified)
        # Detect state variables
        state_pattern = r'(?:state|status|mode|step|stage|phase)\s*=\s*[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?'
        state_matches = re.findall(state_pattern, input_content)
        
        for state_match in state_matches:
            if state_match not in states:
                states.append(state_match)
        
        # Detect functions/methods
        function_pattern = r'(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
        functions = re.findall(function_pattern, input_content)
        
        # Each function becomes an action
        for func_name, params in functions:
            if not func_name.startswith('_'):  # Skip private functions
                actions.append(func_name.replace('_', ' '))
        
        # Detect state changes within functions
        state_change_pattern = r'(?:self\.)?state\s*=\s*[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?'
        
        for func_name, params in functions:
            if func_name.startswith('_'):
                continue
                
            # Find the function body
            func_start = input_content.find(f"def {func_name}")
            if func_start == -1:
                continue
                
            # Find the end of the function (approximate)
            next_def = input_content.find("def ", func_start + 1)
            if next_def != -1:
                func_body = input_content[func_start:next_def]
            else:
                func_body = input_content[func_start:]
            
            # Look for state changes
            state_changes = re.findall(state_change_pattern, func_body)
            
            if state_changes:
                # Function changes state
                action = func_name.replace('_', ' ')
                
                # Use the first detected state as source and last as target
                if len(state_changes) >= 2:
                    source_state = state_changes[0]
                    target_state = state_changes[-1]
                    
                    transitions.append({
                        'source': source_state,
                        'target': target_state,
                        'action': action,
                        'guard': '',
                        'expected': f"Function {func_name} changes state from {source_state} to {target_state}"
                    })
                else:
                    # Only one state change detected
                    target_state = state_changes[0]
                    
                    # Try to infer a source state
                    for state in states:
                        if state != target_state:
                            transitions.append({
                                'source': state,
                                'target': target_state,
                                'action': action,
                                'guard': '',
                                'expected': f"Function {func_name} changes state to {target_state}"
                            })
                            break
    
    # If no states found, use default states
    if not states:
        states = ["Initial", "Processing", "Completed", "Error"]
    
    # If no actions found, use default actions
    if not actions:
        actions = ["Submit", "Cancel", "Save", "Edit", "Delete", "View"]
    
    # If no transitions found, create default transitions
    if not transitions:
        # Create basic transitions between consecutive states
        for i in range(len(states) - 1):
            source = states[i]
            target = states[i + 1]
            action = actions[i % len(actions)]
            
            transitions.append({
                'source': source,
                'target': target,
                'action': action,
                'guard': '',
                'expected': f"System transitions from {source} to {target}"
            })
            
        # Add some cycle transitions
        if len(states) > 1:
            transitions.append({
                'source': states[-1],
                'target': states[0],
                'action': actions[-1],
                'guard': '',
                'expected': f"System cycles back to {states[0]}"
            })
    
    return states, actions, transitions

def _build_state_model(states: List[str], transitions: List[Dict[str, str]]) -> nx.DiGraph:
    """
    Build a directed graph representing the state model.
    
    Args:
        states: List of system states
        transitions: List of state transitions
        
    Returns:
        NetworkX DiGraph representing the state model
    """
    G = nx.DiGraph()
    
    # Add all states as nodes
    for state in states:
        G.add_node(state)
    
    # Add transitions as edges
    for transition in transitions:
        source = transition['source']
        target = transition['target']
        
        # Skip invalid transitions
        if source not in states or target not in states:
            continue
            
        G.add_edge(source, target, 
                   action=transition['action'],
                   guard=transition.get('guard', ''),
                   expected=transition.get('expected', f"System transitions from {source} to {target}"))
    
    return G

def _generate_test_cases_from_model(state_model: nx.DiGraph, actions: List[str], transitions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Generate test cases from the state model.
    
    Args:
        state_model: NetworkX DiGraph representing the state model
        actions: List of all possible actions
        transitions: List of all transitions
        
    Returns:
        List of test cases
    """
    test_cases = []
    
    # 1. State coverage - test each state
    for state in state_model.nodes():
        test_case = {
            'name': f"Verify {state} state",
            'description': f"Test that the system can reach and verify the {state} state",
            'preconditions': "System is operational",
            'expected_result': f"System successfully enters the {state} state",
            'steps': [
                f"Initialize the system",
                f"Navigate to the {state} state",
                f"Verify that the system is in the {state} state"
            ]
        }
        test_cases.append(test_case)
    
    # 2. Transition coverage - test each transition
    for source, target, data in state_model.edges(data=True):
        action = data.get('action', 'perform action')
        expected = data.get('expected', f"System transitions from {source} to {target}")
        guard = data.get('guard', '')
        
        steps = []
        steps.append(f"Ensure the system is in the '{source}' state")
        
        if guard:
            steps.append(f"Ensure condition: {guard}")
            
        steps.append(f"Perform the '{action}' action")
        steps.append(f"Verify that the system transitions to the '{target}' state")
        
        test_case = {
            'name': f"{action.title()} from {source} state",
            'description': f"Test the '{action}' action when starting from the '{source}' state",
            'preconditions': f"System is in the '{source}' state",
            'expected_result': expected,
            'steps': steps
        }
        test_cases.append(test_case)
    
    # 3. Basic path coverage - test a basic workflow through the system
    if len(list(state_model.nodes())) > 1:
        # Try to find a path through all states
        try:
            # Get a topological sort if possible (for DAGs)
            path = list(nx.topological_sort(state_model))
        except nx.NetworkXUnfeasible:
            # If not a DAG, find a simple path from first to last state
            all_states = list(state_model.nodes())
            try:
                path = nx.shortest_path(state_model, all_states[0], all_states[-1])
            except nx.NetworkXNoPath:
                # If no path exists, just use the states in order
                path = all_states
        
        steps = []
        steps.append(f"Initialize the system to its starting state")
        
        # Create steps for each transition in the path
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Find the transition data
            edge_data = state_model.get_edge_data(source, target)
            if edge_data:
                action = edge_data.get('action', 'perform action')
                steps.append(f"In the '{source}' state, perform the '{action}' action")
                steps.append(f"Verify transition to the '{target}' state")
            else:
                steps.append(f"Navigate from '{source}' state to '{target}' state")
        
        test_case = {
            'name': "Complete system workflow test",
            'description': "Test a complete workflow through the system states",
            'preconditions': "System is in its initial state",
            'expected_result': "System successfully completes the entire workflow",
            'steps': steps
        }
        test_cases.append(test_case)
    
    # 4. Invalid transitions test
    if len(list(state_model.nodes())) > 1 and len(actions) > 0:
        all_states = list(state_model.nodes())
        all_actions = list(actions)
        
        for i in range(min(3, len(all_states))):  # Limit to 3 states
            state = all_states[i]
            
            # Find actions that are not defined for this state
            valid_actions = set()
            for _, target, data in state_model.out_edges(state, data=True):
                if 'action' in data:
                    valid_actions.add(data['action'])
            
            invalid_actions = [a for a in all_actions if a not in valid_actions]
            
            if invalid_actions:
                test_case = {
                    'name': f"Invalid actions in {state} state",
                    'description': f"Test that invalid actions in the {state} state are handled properly",
                    'preconditions': f"System is in the {state} state",
                    'expected_result': "System handles invalid actions gracefully without unexpected state changes",
                    'steps': [
                        f"Ensure the system is in the '{state}' state",
                        f"Attempt to perform the invalid action '{invalid_actions[0]}'",
                        f"Verify that the system remains in the '{state}' state",
                        f"Verify that an appropriate error message is displayed"
                    ]
                }
                test_cases.append(test_case)
    
    return test_cases