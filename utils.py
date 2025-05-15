import logging
import re
import random
import nltk
from typing import List, Dict, Any

# Import test generators - direct import of modules to use native functions
from test_generators import model_based, mutation, reinforcement, behavior_driven, fuzz

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

def generate_test_cases(input_content, test_type, input_type):
    """
    Generate test cases based on the input content and test type.
    
    Args:
        input_content: The input content (requirements or code)
        test_type: The type of test to generate (model-based, mutation, reinforcement, behavior-driven, fuzz)
        input_type: The type of input (requirements or code)
        
    Returns:
        list: List of test cases
    """
    try:
        if test_type == 'model-based':
            test_cases = model_based.generate_tests(input_type, input_content)
        elif test_type == 'mutation':
            test_cases = mutation.generate_mutation_tests(input_type, input_content)
        elif test_type == 'reinforcement':
            test_cases = reinforcement.generate_reinforcement_tests(input_type, input_content)
        elif test_type == 'behavior-driven':
            test_cases = behavior_driven.generate_behavior_driven_tests(input_type, input_content)
        elif test_type == 'fuzz':
            test_cases = fuzz.generate_fuzz_tests(input_type, input_content)
        else:
            logger.error(f"Unsupported test type: {test_type}")
            test_cases = []
            
        # Ensure all test cases have the required fields
        standardized_test_cases = []
        for tc in test_cases:
            standardized_tc = {
                'name': tc.get('name', 'Untitled Test Case'),
                'description': tc.get('description', 'No description provided'),
                'preconditions': tc.get('preconditions', tc.get('setup', '')),  # Use 'preconditions' field, fall back to 'setup'
                'expected_result': tc.get('expected_result', ''),
                'steps': tc.get('steps', [])
            }
            standardized_test_cases.append(standardized_tc)
            
        return standardized_test_cases
    except Exception as e:
        logger.exception(f"Error generating {test_type} test cases: {str(e)}")
        return []

def preprocess_text(text):
    """Preprocess text for analysis."""
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s\.]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text, top_n=20):
    """Extract important keywords from text."""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Tokenize
    tokens = word_tokenize(processed_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Count frequency
    word_counts = Counter(filtered_tokens)
    
    # Extract multi-word phrases
    bigrams = list(nltk.bigrams(filtered_tokens))
    bigram_counts = Counter(bigrams)
    
    # Rank keywords (single words and bigrams)
    top_words = [word for word, count in word_counts.most_common(top_n)]
    
    top_bigrams = []
    for (w1, w2), count in bigram_counts.most_common(top_n // 2):
        # Only consider bigrams with sufficient frequency
        if count > 1:
            bigram = f"{w1} {w2}"
            top_bigrams.append(bigram)
    
    # Combine single words and bigrams
    all_keywords = top_words + top_bigrams
    
    # Remove duplicates and limit to top_n
    seen = set()
    unique_keywords = []
    for keyword in all_keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords[:top_n]

#===========================================
# Model-Based Testing Implementation
#===========================================

def generate_model_based_tests(input_content, input_type):
    """Generate model-based tests from requirements or code."""
    test_cases = []
    
    # Extract states, actions, and transitions
    states, actions, transitions = extract_model_components(input_content, input_type)
    
    # Create test cases from the extracted model
    for i, transition in enumerate(transitions[:10]):  # Limit to 10 transitions
        source_state = transition.get('source', 'Initial State')
        target_state = transition.get('target', 'Result State')
        action = transition.get('action', 'Perform Action')
        condition = transition.get('condition', '')
        expected = transition.get('expected', 'Action completes successfully')
        
        # Create steps
        steps = []
        if condition:
            steps.append(f"Ensure the system is in the '{source_state}' state with {condition}")
        else:
            steps.append(f"Ensure the system is in the '{source_state}' state")
            
        steps.append(f"Perform the '{action}' action")
        
        # Add verification step
        steps.append(f"Verify that the system transitions to the '{target_state}' state")
        
        test_case = {
            'name': f"{action.title()} in {source_state} state",
            'description': f"Test the '{action}' action when starting from the '{source_state}' state",
            'preconditions': f"System is in the '{source_state}' state",
            'expected_result': expected,
            'steps': steps
        }
        
        test_cases.append(test_case)
    
    # If no transitions were identified, create some basic tests
    if not test_cases:
        basic_tests = generate_basic_tests(input_content, input_type)
        test_cases.extend(basic_tests)
        
    return test_cases

def extract_model_components(input_content, input_type):
    """Extract states, actions, and transitions from input content."""
    states = []
    actions = []
    transitions = []
    
    # Process differently based on input type
    if input_type == 'requirements':
        # Extract from requirements text
        sentences = nltk.sent_tokenize(input_content)
        
        # Extract potential states
        state_patterns = [
            r'(?:state|screen|page|view)\s+["\']?([A-Za-z0-9_\s]+)["\']?',
            r'(?:in|at|on)\s+(?:the)?\s*([A-Za-z0-9_\s]+)\s+(?:state|screen|page|view)',
            r'(?:system|user)\s+(?:is|should be)\s+(?:in|at)\s+(?:the)?\s*([A-Za-z0-9_\s]+)'
        ]
        
        for sentence in sentences:
            for pattern in state_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    state = match.strip()
                    if state and state not in states:
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
                    if action and action not in actions:
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
                            'condition': '',
                            'expected': f"System transitions from {source_state} to {target_state}"
                        })
                    elif source_state:
                        # Infer a generic target state
                        target_state = "Result State"
                        transitions.append({
                            'source': source_state,
                            'target': target_state,
                            'action': action,
                            'condition': '',
                            'expected': f"Action {action} completes successfully"
                        })
                    
    elif input_type == 'code':
        # Extract from source code (simplified)
        # Detect functions/methods
        function_pattern = r'(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
        functions = re.findall(function_pattern, input_content)
        
        # Each function becomes an action
        for func_name, params in functions:
            if not func_name.startswith('_'):  # Skip private functions
                actions.append(func_name.replace('_', ' '))
        
        # Detect potential states
        state_pattern = r'(?:state|status|mode|step|stage|phase)\s*=\s*[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?'
        state_matches = re.findall(state_pattern, input_content)
        
        for state_match in state_matches:
            states.append(state_match)
        
        # If no states found, infer from code structure
        if not states:
            states = ["Initial", "Processing", "Completed", "Error"]
        
        # Create transitions based on detected functions
        for action in actions:
            transitions.append({
                'source': "Initial",
                'target': "Completed",
                'action': action,
                'condition': '',
                'expected': f"Function {action} executes successfully"
            })
    
    # If still no states, actions, or transitions, create defaults
    if not states:
        states = ["Initial", "Processing", "Completed", "Error"]
    
    if not actions:
        actions = ["Submit", "Create", "Update", "Delete", "View"]
    
    if not transitions:
        # Create basic transitions
        for i in range(len(states) - 1):
            source = states[i]
            target = states[i + 1]
            action = actions[i % len(actions)]
            
            transitions.append({
                'source': source,
                'target': target,
                'action': action,
                'condition': '',
                'expected': f"System transitions from {source} to {target}"
            })
    
    return states, actions, transitions

#===========================================
# Mutation Testing Implementation
#===========================================

def generate_mutation_tests(input_content, input_type):
    """Generate mutation tests from requirements or code."""
    test_cases = []
    
    # Identify mutation points
    mutation_points = identify_mutation_points(input_content, input_type)
    
    # Generate test cases for each mutation point
    for i, point in enumerate(mutation_points[:10]):  # Limit to 10 points
        if input_type == 'code':
            # For code, focus on mutations to operators and values
            test_case = {
                'name': f"Mutation test for {point['type']} on line {point.get('line', i+1)}",
                'description': f"Test to detect changes in {point['description']}",
                'preconditions': "System implements the original code correctly",
                'expected_result': "The test should fail when the mutation is applied",
                'steps': [
                    f"Apply mutation: {point['description']}",
                    "Execute the affected functionality",
                    "Verify that the system behavior changes as expected"
                ]
            }
        else:
            # For requirements, focus on constraint changes
            test_case = {
                'name': f"Mutation test for requirement constraint",
                'description': f"Test to detect changes in {point['description']}",
                'preconditions': "System implements the requirements correctly",
                'expected_result': "The test should detect violations of the requirement",
                'steps': [
                    f"Modify implementation to violate: {point['description']}",
                    "Execute the affected functionality",
                    "Verify that the test detects the incorrect behavior"
                ]
            }
        
        test_cases.append(test_case)
    
    # If no mutation points were identified, create some basic tests
    if not test_cases:
        basic_tests = generate_basic_tests(input_content, input_type)
        test_cases.extend(basic_tests)
        
    return test_cases

def identify_mutation_points(input_content, input_type):
    """Identify points in the code or requirements where mutations can be applied."""
    mutation_points = []
    
    if input_type == 'code':
        # Analyze code for potential mutation points
        lines = input_content.split('\n')
        
        for i, line in enumerate(lines):
            line_number = i + 1
            
            # Look for comparison operators
            if re.search(r'[<>=!]=|[<>]', line):
                mutation_points.append({
                    'type': 'comparison',
                    'line': line_number,
                    'description': f"Change comparison operator on line {line_number}"
                })
            
            # Look for arithmetic operators
            if re.search(r'[+\-*/%]', line):
                mutation_points.append({
                    'type': 'arithmetic',
                    'line': line_number,
                    'description': f"Change arithmetic operator on line {line_number}"
                })
            
            # Look for boolean operators
            if re.search(r'\b(and|or|not|&&|\|\|)\b', line):
                mutation_points.append({
                    'type': 'boolean',
                    'line': line_number,
                    'description': f"Change boolean operator on line {line_number}"
                })
            
            # Look for return statements
            if re.search(r'\breturn\b', line):
                mutation_points.append({
                    'type': 'return',
                    'line': line_number,
                    'description': f"Modify return value on line {line_number}"
                })
            
            # Look for numeric literals
            for match in re.finditer(r'\b\d+\b', line):
                mutation_points.append({
                    'type': 'constant',
                    'line': line_number,
                    'description': f"Change numeric constant {match.group(0)} on line {line_number}"
                })
    else:
        # Analyze requirements for constraints that could be mutated
        sentences = nltk.sent_tokenize(input_content)
        
        # Keywords that often indicate constraints or business rules
        constraint_keywords = [
            'must', 'should', 'shall', 'required', 'always', 'never', 'only',
            'at least', 'at most', 'greater than', 'less than', 'equal to',
            'maximum', 'minimum', 'between', 'valid', 'invalid'
        ]
        
        for i, sentence in enumerate(sentences):
            for keyword in constraint_keywords:
                if keyword in sentence.lower():
                    mutation_points.append({
                        'type': 'requirement',
                        'id': i + 1,
                        'description': f"Modify constraint '{keyword}' in requirement: {sentence.strip()}"
                    })
                    break
        
        # Extract numeric constraints
        numeric_pattern = r'(\d+)\s*(second|minute|hour|day|week|month|year|percent|%|item|user|time)'
        for i, sentence in enumerate(sentences):
            matches = re.findall(numeric_pattern, sentence, re.IGNORECASE)
            for match in matches:
                value, unit = match
                mutation_points.append({
                    'type': 'numeric_constraint',
                    'id': i + 1,
                    'description': f"Change numeric value '{value} {unit}' in requirement: {sentence.strip()}"
                })
    
    return mutation_points

#===========================================
# Reinforcement Learning Testing Implementation
#===========================================

def generate_reinforcement_tests(input_content, input_type):
    """Generate reinforcement learning-based tests from requirements or code."""
    test_cases = []
    
    # Extract key components
    components = extract_reinforcement_components(input_content, input_type)
    
    # Generate test cases based on the extracted components
    actions = components.get('actions', [])
    states = components.get('states', [])
    conditions = components.get('conditions', [])
    
    # Generate test cases with comprehensive test paths
    for i, action in enumerate(actions[:5]):  # Limit to 5 actions
        # Create a basic test path
        if states:
            # Find a path through states
            path_length = min(4, len(states))
            state_path = random.sample(states, path_length)
            
            steps = []
            steps.append(f"Ensure the system is in the '{state_path[0]}' state")
            
            for j in range(path_length - 1):
                steps.append(f"Perform the '{action}' action")
                steps.append(f"Verify the system transitions to the '{state_path[j+1]}' state")
                
                # Add a condition check if available
                if conditions and j < len(conditions):
                    steps.append(f"Verify that '{conditions[j]}' is satisfied")
            
            test_case = {
                'name': f"Path test for {action}",
                'description': f"Test a sequence of states and transitions using the '{action}' action",
                'preconditions': f"System is operational and in the '{state_path[0]}' state",
                'expected_result': f"The system successfully traverses all states in the path",
                'steps': steps
            }
            
            test_cases.append(test_case)
        else:
            # Create a basic action test
            steps = [
                "Ensure the system is operational",
                f"Perform the '{action}' action",
                "Verify the action completes successfully"
            ]
            
            test_case = {
                'name': f"Basic test for {action}",
                'description': f"Test the '{action}' action under normal conditions",
                'preconditions': "System is operational",
                'expected_result': f"The '{action}' action completes successfully",
                'steps': steps
            }
            
            test_cases.append(test_case)
    
    # Edge case tests
    if states:
        for i, state in enumerate(states[:3]):  # Limit to 3 states
            steps = [
                f"Ensure the system is in the '{state}' state",
                "Attempt to perform an invalid operation",
                "Verify the system handles the error appropriately"
            ]
            
            test_case = {
                'name': f"Edge case test in {state} state",
                'description': f"Test system behavior when invalid operations are attempted in the '{state}' state",
                'preconditions': f"System is in the '{state}' state",
                'expected_result': "The system handles the error gracefully without crashing",
                'steps': steps
            }
            
            test_cases.append(test_case)
    
    # If no tests were created, generate some basic tests
    if not test_cases:
        basic_tests = generate_basic_tests(input_content, input_type)
        test_cases.extend(basic_tests)
        
    return test_cases

def extract_reinforcement_components(input_content, input_type):
    """Extract components needed for reinforcement learning-based testing."""
    components = {
        'actions': [],
        'states': [],
        'conditions': []
    }
    
    if input_type == 'requirements':
        # Extract from requirements text
        sentences = nltk.sent_tokenize(input_content)
        
        # Extract actions
        action_pattern = r'\b(create|read|update|delete|submit|send|receive|process|validate|check|login|logout|register|add|remove|edit|view|search|filter|sort|calculate|generate|print|export|import|upload|download)\b\s+\w+'
        action_matches = re.findall(action_pattern, input_content, re.IGNORECASE)
        components['actions'] = list(set(action_matches))
        
        # Use POS tagging to find more actions (verbs)
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            
            # Extract verbs
            verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
            components['actions'].extend(verbs)
        
        # Remove duplicates
        components['actions'] = list(set(components['actions']))
        
        # Extract states
        state_pattern = r'\b(initial|start|begin|logged in|authenticated|selected|completed|finished|error|exception|invalid|valid|success|failure)\b'
        state_matches = re.findall(state_pattern, input_content, re.IGNORECASE)
        components['states'] = list(set(state_matches))
        
        # Extract conditions
        condition_pattern = r'\b(if|when|must|should|only if|unless|until|after|before|while)\b'
        for i, sentence in enumerate(sentences):
            if re.search(condition_pattern, sentence, re.IGNORECASE):
                components['conditions'].append(sentence.strip())
    
    elif input_type == 'code':
        # Extract from source code
        
        # Extract function/method names as actions
        function_pattern = r'(def|function)\s+([A-Za-z0-9_]+)\s*\('
        functions = re.findall(function_pattern, input_content)
        function_names = [func[1] for func in functions]
        
        # Clean function names
        for func in function_names:
            # Convert snake_case or camelCase to spaces
            readable_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', func)  # camelCase
            readable_name = readable_name.replace('_', ' ')  # snake_case
            components['actions'].append(readable_name)
        
        # Extract conditional statements as possible states
        if_pattern = r'if\s+(.+?):'
        if_conditions = re.findall(if_pattern, input_content)
        components['conditions'] = if_conditions
        
        # Extract potential state variables
        state_var_pattern = r'(state|status|mode|phase|step|stage|condition)\s*='
        state_vars = re.findall(state_var_pattern, input_content, re.IGNORECASE)
        components['states'].extend(state_vars)
        
        # Add default states if none found
        if not components['states']:
            components['states'] = ['initial', 'processing', 'complete', 'error']
    
    # Ensure we have some defaults if extraction failed
    if not components['actions']:
        components['actions'] = ['create', 'read', 'update', 'delete', 'submit', 'validate']
    
    if not components['states']:
        components['states'] = ['initial', 'processing', 'completed', 'error']
    
    return components

#===========================================
# Behavior-Driven Testing Implementation
#===========================================

def generate_behavior_driven_tests(input_content, input_type):
    """Generate behavior-driven tests from requirements or code."""
    test_cases = []
    
    if input_type == 'requirements':
        # Process requirements
        scenarios = extract_bdd_scenarios(input_content)
        for i, scenario in enumerate(scenarios[:5]):  # Limit to 5 scenarios
            steps = []
            
            # Given step
            if scenario.get('given'):
                steps.append(f"Given {scenario.get('given')}")
            else:
                steps.append("Given the system is ready")
            
            # When step
            if scenario.get('when'):
                steps.append(f"When {scenario.get('when')}")
            else:
                steps.append("When the user performs an action")
            
            # Then step
            if scenario.get('then'):
                steps.append(f"Then {scenario.get('then')}")
            else:
                steps.append("Then the system responds correctly")
            
            test_case = {
                'name': scenario.get('name', f"BDD Scenario {i+1}"),
                'description': scenario.get('description', f"Behavior-driven test scenario {i+1}"),
                'preconditions': scenario.get('given', "System is ready"),
                'expected_result': scenario.get('then', "The system responds as expected"),
                'steps': steps
            }
            test_cases.append(test_case)
    
    elif input_type == 'code':
        # Process source code
        code_scenarios = extract_bdd_scenarios_from_code(input_content)
        for i, scenario in enumerate(code_scenarios[:5]):  # Limit to 5 scenarios
            steps = []
            
            # Given step
            if scenario.get('given'):
                steps.append(f"Given {scenario.get('given')}")
            else:
                steps.append("Given the function preconditions are met")
            
            # When step
            if scenario.get('when'):
                steps.append(f"When {scenario.get('when')}")
            else:
                steps.append("When the function is called")
            
            # Then step
            if scenario.get('then'):
                steps.append(f"Then {scenario.get('then')}")
            else:
                steps.append("Then the function returns the expected result")
            
            test_case = {
                'name': scenario.get('name', f"Code Behavior Test {i+1}"),
                'description': scenario.get('description', f"Behavior-driven test for code function {i+1}"),
                'preconditions': scenario.get('given', "Function preconditions are met"),
                'expected_result': scenario.get('then', "Function returns expected result"),
                'steps': steps
            }
            test_cases.append(test_case)
    
    # If no scenarios were found, create some basic tests
    if not test_cases:
        basic_tests = generate_basic_tests(input_content, input_type)
        test_cases.extend(basic_tests)
    
    return test_cases

def extract_bdd_scenarios(requirements):
    """Extract behavior-driven scenarios from requirements text."""
    scenarios = []
    
    # Look for explicit BDD scenarios (Gherkin)
    gherkin_pattern = r'(?:Scenario|Feature)[:]\s*([^\n]+)(?:\n|\r\n?)+\s*Given\s+([^\n]+)(?:\n|\r\n?)+\s*When\s+([^\n]+)(?:\n|\r\n?)+\s*Then\s+([^\n]+)'
    gherkin_matches = re.findall(gherkin_pattern, requirements, re.IGNORECASE | re.MULTILINE)
    
    for match in gherkin_matches:
        name, given, when, then = match
        scenarios.append({
            'name': name.strip(),
            'description': f"Scenario: {name.strip()}",
            'given': given.strip(),
            'when': when.strip(),
            'then': then.strip()
        })
    
    # If no explicit Gherkin, look for implicit scenarios
    if not scenarios:
        implicit_scenarios = _extract_implicit_scenarios(requirements)
        scenarios.extend(implicit_scenarios)
    
    return scenarios

def _extract_implicit_scenarios(text):
    """Extract implicit scenarios from text where structured BDD formats aren't present."""
    scenarios = []
    
    # Preprocess to normalize line endings and remove excessive whitespace
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    
    # Try to identify user stories and convert to scenarios
    user_story_pattern = r'As (?:a|an) ([^,]+),\s*I want to ([^,]+)(?:,?\s*so that ([^\.]+))?'
    user_stories = re.findall(user_story_pattern, text, re.IGNORECASE)
    
    for i, story in enumerate(user_stories):
        role, action, benefit = story
        
        # Form BDD scenario from user story
        given = f"I am a {role.strip()}"
        when = f"I {action.strip()}"
        then = benefit.strip() if benefit else f"I complete the action successfully"
        
        scenarios.append({
            'name': f"User Story {i+1}",
            'description': f"As a {role.strip()}, I want to {action.strip()}" + 
                          (f", so that {benefit.strip()}" if benefit else ""),
            'given': given,
            'when': when,
            'then': then
        })
    
    # If still no scenarios, try to infer from normal text
    if not scenarios:
        sentences = nltk.sent_tokenize(text)
        
        # Group sentences into potential scenarios (3 sentences per scenario)
        for i in range(0, len(sentences), 3):
            if i + 2 < len(sentences):
                name = f"Inferred Scenario {i//3 + 1}"
                
                # Analyze sentence structure to determine best match for given/when/then
                tokens1 = nltk.word_tokenize(sentences[i])
                pos1 = nltk.pos_tag(tokens1)
                tokens2 = nltk.word_tokenize(sentences[i+1])
                pos2 = nltk.pos_tag(tokens2)
                tokens3 = nltk.word_tokenize(sentences[i+2])
                pos3 = nltk.pos_tag(tokens3)
                
                # Check if first sentence is a condition (Given)
                is_condition1 = any(word.lower() in ["if", "when", "while", "assuming", "given"] for word, _ in pos1[:3])
                # Check if second sentence has an action verb (When)
                has_action2 = any(pos.startswith('VB') for _, pos in pos2[:5])
                # Check if third sentence has result indicators (Then)
                has_result3 = any(word.lower() in ["should", "must", "will", "then", "expects", "results"] for word, _ in pos3[:5])
                
                if is_condition1 and has_action2 and has_result3:
                    # Good match for Given-When-Then
                    given = sentences[i]
                    when = sentences[i+1]
                    then = sentences[i+2]
                elif has_action2:
                    # If second sentence has action verb, use default for Given
                    given = "The system is ready"
                    when = sentences[i+1]
                    then = sentences[i+2] if i+2 < len(sentences) else "The system responds correctly"
                else:
                    # Default fallback
                    given = sentences[i]
                    when = sentences[i+1] if i+1 < len(sentences) else "The user performs an action"
                    then = sentences[i+2] if i+2 < len(sentences) else "The system responds correctly"
                
                scenarios.append({
                    'name': name,
                    'description': f"{given} {when} {then}",
                    'given': given,
                    'when': when,
                    'then': then
                })
    
    return scenarios

def extract_bdd_scenarios_from_code(code):
    """Extract behavior-driven scenarios from source code."""
    scenarios = []
    
    # Look for function/method definitions
    function_pattern = r'(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)[^{]*(?:{|\:)'
    functions = re.findall(function_pattern, code, re.DOTALL)
    
    for i, (func_name, params) in enumerate(functions):
        # Skip private/helper functions
        if func_name.startswith('_'):
            continue
        
        # Convert snake_case or camelCase to spaces for readability
        readable_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', func_name)  # camelCase
        readable_name = readable_name.replace('_', ' ')  # snake_case
        
        # Parse parameters
        param_list = params.split(',')
        clean_params = []
        for param in param_list:
            param = param.strip()
            if param:
                # Remove type hints/annotations
                param_name = param.split(':')[0].strip() if ':' in param else param
                param_name = param_name.split('=')[0].strip() if '=' in param_name else param_name
                clean_params.append(param_name)
        
        # Construct a scenario based on function signature
        given = f"the function {readable_name} is called"
        
        if clean_params:
            param_str = ", ".join(clean_params)
            when = f"the parameters are provided ({param_str})"
        else:
            when = f"the function is called with valid input"
        
        # Look for return statements in function body
        # Extract the function body
        function_body_pattern = r'def\s+' + re.escape(func_name) + r'\s*\(.*?\).*?:(.+?)(?=\n\w+|\n\s*def|\Z)'
        body_match = re.search(function_body_pattern, code, re.DOTALL)
        
        then = "the function returns the expected result"
        if body_match:
            body = body_match.group(1)
            
            # Check for return statements
            return_pattern = r'return\s+(.+)'
            returns = re.findall(return_pattern, body)
            
            if returns:
                # Use the last return statement for the Then part
                then = f"the function returns {returns[-1].strip()}"
            
            # Check for exceptions/error handling
            if 'except' in body or 'try' in body:
                then += " or handles errors appropriately"
        
        scenarios.append({
            'name': f"Function: {readable_name}",
            'description': f"Test behavior of the {readable_name} function",
            'given': given,
            'when': when,
            'then': then
        })
    
    return scenarios


#===========================================
# Fuzz Testing Implementation
#===========================================

def generate_fuzz_tests(input_content, input_type):
    """Generate fuzz tests from requirements or code."""
    test_cases = []
    
    if input_type == 'code':
        # Extract functions to fuzz
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
        functions = re.findall(function_pattern, input_content)
        
        for i, (func_name, params) in enumerate(functions[:3]):  # Limit to 3 functions
            # Skip private functions
            if func_name.startswith('_'):
                continue
                
            # Parse parameters
            param_list = []
            if params:
                for p in params.split(','):
                    p = p.strip()
                    if p:
                        # Extract parameter name (remove type annotations and default values)
                        if '=' in p:
                            p = p.split('=')[0].strip()
                        if ':' in p:
                            p = p.split(':')[0].strip()
                        param_list.append(p)
            
            # Create different types of fuzz tests for this function
            
            # 1. Random input test
            steps_random = [
                f"Prepare random inputs for function {func_name}",
                f"Call function with completely random values",
                "Verify the function handles unexpected input gracefully"
            ]
            
            test_case_random = {
                'name': f"Random input test for {func_name}",
                'description': f"Test {func_name} with completely random input values to check for crashes",
                'preconditions': "Function is accessible",
                'expected_result': "Function should not crash and should handle invalid inputs gracefully",
                'steps': steps_random
            }
            test_cases.append(test_case_random)
            
            # 2. Boundary value test
            steps_boundary = [
                f"Prepare boundary values for function {func_name}",
                f"Call function with extreme values (e.g., empty values, very large numbers, special characters)",
                "Verify the function handles boundary cases correctly"
            ]
            
            test_case_boundary = {
                'name': f"Boundary value test for {func_name}",
                'description': f"Test {func_name} with boundary/edge values to check for proper handling",
                'preconditions': "Function is accessible",
                'expected_result': "Function should handle boundary cases according to specifications",
                'steps': steps_boundary
            }
            test_cases.append(test_case_boundary)
            
            # 3. Malformed input test
            steps_malformed = [
                f"Prepare malformed inputs for function {func_name} (wrong types, null values, etc.)",
                f"Call function with incompatible input types",
                "Verify the function validates input types and provides appropriate errors"
            ]
            
            test_case_malformed = {
                'name': f"Malformed input test for {func_name}",
                'description': f"Test {func_name} with inputs of incorrect types or formats",
                'preconditions': "Function is accessible",
                'expected_result': "Function should validate inputs and fail gracefully with clear error messages",
                'steps': steps_malformed
            }
            test_cases.append(test_case_malformed)
    
    elif input_type == 'requirements':
        # Extract input fields, data types and constraints from requirements
        input_fields = _extract_input_fields_from_requirements(input_content)
        
        # Generate fuzz tests based on extracted fields
        if input_fields:
            for i, field in enumerate(input_fields[:3]):  # Limit to 3 fields
                field_name = field.get('name', f"Field{i+1}")
                
                # 1. Field-specific random value test
                steps_field_random = [
                    f"Identify the '{field_name}' input field",
                    f"Submit random/unexpected values to the field",
                    "Verify the system handles the unexpected input gracefully"
                ]
                
                test_case_field_random = {
                    'name': f"Fuzz test for {field_name} input",
                    'description': f"Test the {field_name} input with random values to check for proper validation",
                    'preconditions': "System is accessible and the input field is available",
                    'expected_result': "System should validate the input and handle invalid data gracefully",
                    'steps': steps_field_random
                }
                test_cases.append(test_case_field_random)
                
                # 2. Field-specific boundary test
                steps_field_boundary = [
                    f"Identify the '{field_name}' input field",
                    f"Submit boundary values to the field (e.g., empty, maximum length, special characters)",
                    "Verify the system correctly validates the boundary values"
                ]
                
                test_case_field_boundary = {
                    'name': f"Boundary fuzz test for {field_name}",
                    'description': f"Test the {field_name} input with boundary values to check for proper validation",
                    'preconditions': "System is accessible and the input field is available",
                    'expected_result': "System should correctly validate boundary values according to requirements",
                    'steps': steps_field_boundary
                }
                test_cases.append(test_case_field_boundary)
        
        # Generic fuzz tests for the system
        steps_workflow = [
            "Identify a critical workflow in the system",
            "Perform random operations in an unexpected sequence",
            "Verify the system maintains data integrity and handles unexpected workflows gracefully"
        ]
        
        test_case_workflow = {
            'name': "Workflow fuzzing test",
            'description': "Test the system with random workflow steps and unexpected sequences",
            'preconditions': "System is operational",
            'expected_result': "System should maintain stability and data integrity when exposed to unexpected workflows",
            'steps': steps_workflow
        }
        test_cases.append(test_case_workflow)
        
        # API fuzzing test
        if re.search(r'\b(api|endpoint|service|rest|http|request|response)\b', input_content, re.IGNORECASE):
            steps_api = [
                "Identify API endpoints in the system",
                "Send malformed requests with unexpected headers, parameters, and bodies",
                "Verify the API handles malformed requests safely and returns appropriate error codes"
            ]
            
            test_case_api = {
                'name': "API fuzzing test",
                'description': "Test API endpoints with malformed requests to check for security issues",
                'preconditions': "API endpoints are accessible",
                'expected_result': "API should validate all inputs and respond with appropriate error codes for invalid requests",
                'steps': steps_api
            }
            test_cases.append(test_case_api)
    
    # If no tests were generated, create some generic tests
    if not test_cases:
        # Generic fuzz test
        test_case_generic = {
            'name': "Generic fuzz test",
            'description': "Test the system with random and unexpected inputs to check for robustness",
            'preconditions': "System is operational",
            'expected_result': "System should handle unexpected inputs gracefully without crashing",
            'steps': [
                "Identify input fields in the system",
                "Submit random, malformed, and boundary values to these inputs",
                "Verify the system handles the unexpected inputs gracefully"
            ]
        }
        test_cases.append(test_case_generic)
        
        # Input boundary test
        test_case_boundary = {
            'name': "Input boundary fuzz test",
            'description': "Test the system with boundary values to check for input validation",
            'preconditions': "System is operational",
            'expected_result': "System should validate all inputs according to specifications",
            'steps': [
                "Identify input fields in the system",
                "Submit boundary values (empty, maximum length, special characters, etc.)",
                "Verify the system correctly validates and processes these values"
            ]
        }
        test_cases.append(test_case_boundary)
        
        # Sequence fuzzing test
        test_case_sequence = {
            'name': "Operation sequence fuzz test",
            'description': "Test the system with random operation sequences to check for state corruption",
            'preconditions': "System is operational",
            'expected_result': "System should maintain data integrity when operations are performed in unexpected sequences",
            'steps': [
                "Identify a series of operations in the system",
                "Perform these operations in random and unexpected sequences",
                "Verify the system maintains data integrity and does not enter an inconsistent state"
            ]
        }
        test_cases.append(test_case_sequence)
    
    return test_cases

def _extract_input_fields_from_requirements(requirements):
    """Extract input fields from requirements text."""
    fields = []
    
    # Look for input field descriptions
    field_patterns = [
        r'(?:input|field|box|entry|form)\s+(?:for|called|named)?\s*[\'"]?([a-zA-Z0-9_\s]+)[\'"]?',
        r'(?:enter|input|type|fill in)\s+(?:the|a|an)\s+([a-zA-Z0-9_\s]+)',
        r'([a-zA-Z0-9_\s]+)\s+(?:field|input|box|entry)'
    ]
    
    for pattern in field_patterns:
        matches = re.findall(pattern, requirements, re.IGNORECASE)
        for match in matches:
            field_name = match.strip()
            
            # Avoid duplicates and very short field names
            if field_name and len(field_name) > 2 and not any(f['name'] == field_name for f in fields):
                field = {'name': field_name}
                
                # Try to determine field type
                if re.search(rf'{field_name}.*?(?:number|integer|numeric|float|decimal)', requirements, re.IGNORECASE):
                    field['type'] = 'numeric'
                elif re.search(rf'{field_name}.*?(?:date|time|calendar)', requirements, re.IGNORECASE):
                    field['type'] = 'date/time'
                elif re.search(rf'{field_name}.*?(?:email|e-mail)', requirements, re.IGNORECASE):
                    field['type'] = 'email'
                elif re.search(rf'{field_name}.*?(?:password|credential)', requirements, re.IGNORECASE):
                    field['type'] = 'password'
                elif re.search(rf'{field_name}.*?(?:select|dropdown|drop-down|menu|option)', requirements, re.IGNORECASE):
                    field['type'] = 'selection'
                elif re.search(rf'{field_name}.*?(?:checkbox|check box|toggle|switch)', requirements, re.IGNORECASE):
                    field['type'] = 'boolean'
                else:
                    field['type'] = 'text'
                
                # Look for constraints
                constraints = []
                
                # Required
                if re.search(rf'{field_name}.*?(?:required|mandatory|must)', requirements, re.IGNORECASE):
                    constraints.append('required')
                
                # Length
                length_match = re.search(rf'{field_name}.*?(?:maximum|max|up to).*?(\d+)', requirements, re.IGNORECASE)
                if length_match:
                    constraints.append(f'max_length: {length_match.group(1)}')
                
                min_length_match = re.search(rf'{field_name}.*?(?:minimum|min|at least).*?(\d+)', requirements, re.IGNORECASE)
                if min_length_match:
                    constraints.append(f'min_length: {min_length_match.group(1)}')
                
                field['constraints'] = constraints
                fields.append(field)
    
    return fields

#===========================================
# Fallback Basic Test Generation
#===========================================

def generate_basic_tests(input_content, input_type):
    """Generate basic test cases when specific test generation fails."""
    test_cases = []
    
    # Extract keywords to use in test cases
    keywords = extract_keywords(input_content, 5)
    
    if input_type == 'code':
        # Try to extract function names
        function_pattern = r'(?:def|function)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        functions = re.findall(function_pattern, input_content)
        
        if functions:
            for i, func in enumerate(functions[:3]):
                test_case = {
                    'name': f"Test {func} function",
                    'description': f"Verify the {func} function works correctly",
                    'preconditions': "System is operational",
                    'expected_result': f"Function {func} executes successfully",
                    'steps': [
                        "Prepare test data",
                        f"Call the {func} function",
                        "Verify the function returns expected results"
                    ]
                }
                test_cases.append(test_case)
        
        # Add some generic test cases
        test_cases.append({
            'name': "Test valid input handling",
            'description': "Verify the system correctly processes valid input",
            'preconditions': "System is operational",
            'expected_result': "Valid input is processed successfully",
            'steps': [
                "Prepare valid input data",
                "Submit the data to the system",
                "Verify the system processes the data correctly"
            ]
        })
        
        test_cases.append({
            'name': "Test invalid input handling",
            'description': "Verify the system correctly handles invalid input",
            'preconditions': "System is operational",
            'expected_result': "System rejects invalid input with appropriate error message",
            'steps': [
                "Prepare invalid input data",
                "Submit the data to the system",
                "Verify the system rejects the data with an appropriate error message"
            ]
        })
    else:
        # Requirements-based tests
        for i, keyword in enumerate(keywords):
            if i >= 3:  # Limit to 3 keyword-based tests
                break
                
            test_case = {
                'name': f"Test {keyword.title()} functionality",
                'description': f"Verify the system's {keyword} functionality works correctly",
                'preconditions': "System is operational",
                'expected_result': f"The {keyword} functionality works as expected",
                'steps': [
                    f"Access the {keyword} feature",
                    f"Perform operations related to {keyword}",
                    f"Verify the {keyword} operations complete successfully"
                ]
            }
            test_cases.append(test_case)
        
        # Add some generic test cases
        test_case = {
            'name': "Basic functionality test",
            'description': "Verify the basic system functionality",
            'preconditions': "System is operational",
            'expected_result': "System performs basic operations correctly",
            'steps': [
                "Launch the system",
                "Perform a basic operation",
                "Verify the operation completes successfully"
            ]
        }
        test_cases.append(test_case)
        
        test_case = {
            'name': "Error handling test",
            'description': "Verify the system handles errors appropriately",
            'preconditions': "System is operational",
            'expected_result': "System handles errors gracefully",
            'steps': [
                "Attempt an operation that will cause an error",
                "Verify the system displays an appropriate error message",
                "Verify the system remains stable after the error"
            ]
        }
        test_cases.append(test_case)
    
    return test_cases
