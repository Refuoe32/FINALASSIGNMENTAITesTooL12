"""
Enhanced Behavior-Driven Test Generation

This module implements an advanced Behavior-Driven Development (BDD) test generation system
that creates comprehensive tests verifying system behavior from a user's perspective
using Given-When-Then scenarios with improved pattern recognition and test case variety.
"""
import re
import nltk
from typing import List, Dict, Any, Optional

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def extract_bdd_scenarios(requirements: str) -> List[Dict[str, str]]:
    """
    Extract behavior-driven scenarios from requirements text.

    This function identifies BDD scenarios in different formats (Gherkin, user stories)
    and extracts them into a standardized format.

    Args:
        requirements: String containing the requirements text

    Returns:
        list: List of dictionaries with Given-When-Then scenarios
    """
    scenarios = []
    
    # Look for explicit Gherkin scenarios
    gherkin_pattern = r'(?:Scenario|Feature):\s*(.*?)\n\s*Given\s+(.*?)\n\s*When\s+(.*?)\n\s*Then\s+(.*?)(?:\n|$)'
    for match in re.finditer(gherkin_pattern, requirements, re.IGNORECASE | re.DOTALL):
        scenario = {
            'name': match.group(1).strip(),
            'given': match.group(2).strip(),
            'when': match.group(3).strip(),
            'then': match.group(4).strip()
        }
        scenarios.append(scenario)
    
    # Look for user story format
    user_story_pattern = r'As\s+a\s+(.*?)\s*,?\s*I\s+want\s+to\s+(.*?)\s*,?\s*so\s+that\s+(.*?)(?:\n|$)'
    for match in re.finditer(user_story_pattern, requirements, re.IGNORECASE | re.DOTALL):
        role = match.group(1).strip()
        action = match.group(2).strip()
        benefit = match.group(3).strip()
        
        # Create a sensible scenario from user story
        scenario = {
            'name': f"{role} can {action}",
            'given': f"I am a {role}",
            'when': f"I {action}",
            'then': f"I should {benefit}"
        }
        scenarios.append(scenario)
    
    # Look for implicit scenarios
    if not scenarios:
        implicit_scenarios = _extract_implicit_scenarios(requirements)
        scenarios.extend(implicit_scenarios)
    
    # Create negative scenarios for each positive scenario
    negative_scenarios = []
    for scenario in scenarios:
        negative = _create_negative_scenario(scenario)
        if negative:
            negative_scenarios.append(negative)
    
    # Add negative scenarios
    scenarios.extend(negative_scenarios)
    
    return scenarios

def _extract_implicit_scenarios(text: str) -> List[Dict[str, str]]:
    """
    Extract implicit scenarios from text where structured BDD formats aren't present.

    Args:
        text: Preprocessed text to analyze

    Returns:
        List of scenario dictionaries
    """
    scenarios = []
    
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Extract actions that could be turned into scenarios
    action_patterns = [
        r'(?:user|customer|system|actor)\s+(?:can|should|must|will|may)\s+([a-z]+\s+.*?)(?:\.|\n|$)',
        r'(?:to\s+|ability\s+to\s+)([a-z]+\s+.*?)(?:\.|\n|$)',
        r'(?:when|if)\s+(.*?),\s+(?:then|the system|the user|it)\s+(.*?)(?:\.|\n|$)'
    ]
    
    for sentence in sentences:
        for pattern in action_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                # For when-then pattern
                if 'when' in pattern or 'if' in pattern:
                    condition = match.group(1).strip()
                    result = match.group(2).strip()
                    scenario = {
                        'name': f"System handles {condition}",
                        'given': "The system is operational",
                        'when': condition,
                        'then': result
                    }
                    scenarios.append(scenario)
                else:
                    action = match.group(1).strip()
                    if len(action) > 5:  # Skip very short phrases
                        scenario = {
                            'name': f"User can {action}",
                            'given': "User is logged in",
                            'when': f"User attempts to {action}",
                            'then': f"User successfully {action}s"
                        }
                        scenarios.append(scenario)
    
    # If we still don't have scenarios, create generic ones based on domain-specific keywords
    if not scenarios:
        common_actions = [
            ("view", "viewing information", "sees the requested information"),
            ("create", "creating a new item", "the item is successfully created"),
            ("edit", "editing an item", "the changes are saved successfully"),
            ("delete", "deleting an item", "the item is successfully removed"),
            ("search", "searching for items", "relevant results are displayed"),
            ("filter", "filtering items", "filtered results are displayed"),
            ("sort", "sorting items", "items are displayed in the requested order"),
            ("login", "logging in", "is successfully authenticated"),
            ("register", "registering an account", "account is successfully created"),
            ("upload", "uploading a file", "file is successfully uploaded"),
            ("download", "downloading a file", "file is successfully downloaded")
        ]
        
        for action, when_desc, then_desc in common_actions:
            if action in text.lower():
                scenario = {
                    'name': f"User can {action}",
                    'given': "User has appropriate permissions",
                    'when': f"User performs {when_desc}",
                    'then': f"User {then_desc}"
                }
                scenarios.append(scenario)
    
    return scenarios

def extract_bdd_scenarios_from_code(code: str) -> List[Dict[str, str]]:
    """
    Extract behavior-driven scenarios from source code.

    Args:
        code: String containing the source code

    Returns:
        list: List of dictionaries with Given-When-Then scenarios
    """
    scenarios = []
    
    # Extract functions that can be turned into scenarios
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
    functions = re.findall(function_pattern, code)
    
    for func_name, params in functions:
        # Skip private methods
        if func_name.startswith('_'):
            continue
        
        # Create scenario name from function name
        name = ' '.join(func_name.split('_')).capitalize()
        
        # Extract parameters
        param_list = []
        if params:
            for p in params.split(','):
                p = p.strip()
                if p:
                    # Remove type annotations and default values
                    if '=' in p:
                        p = p.split('=')[0].strip()
                    if ':' in p:
                        p = p.split(':')[0].strip()
                    param_list.append(p)
        
        # Create a scenario based on function purpose (inferred from name)
        if 'get' in func_name or 'fetch' in func_name or 'load' in func_name or 'find' in func_name:
            scenario = {
                'name': f"{name}",
                'given': "The system has the requested data",
                'when': f"The function {func_name} is called with valid parameters",
                'then': "The requested data is returned"
            }
            scenarios.append(scenario)
            
            # Add negative scenario
            scenario_neg = {
                'name': f"{name} with invalid input",
                'given': "The system is operational",
                'when': f"The function {func_name} is called with invalid parameters",
                'then': "An appropriate error is raised"
            }
            scenarios.append(scenario_neg)
        
        elif 'create' in func_name or 'add' in func_name or 'insert' in func_name:
            scenario = {
                'name': f"{name}",
                'given': "The system is ready to accept new data",
                'when': f"The function {func_name} is called with valid data",
                'then': "The new item is successfully created"
            }
            scenarios.append(scenario)
        
        elif 'update' in func_name or 'edit' in func_name or 'modify' in func_name:
            scenario = {
                'name': f"{name}",
                'given': "An existing item needs to be updated",
                'when': f"The function {func_name} is called with updated data",
                'then': "The item is successfully updated"
            }
            scenarios.append(scenario)
        
        elif 'delete' in func_name or 'remove' in func_name:
            scenario = {
                'name': f"{name}",
                'given': "An existing item needs to be removed",
                'when': f"The function {func_name} is called with the item identifier",
                'then': "The item is successfully removed"
            }
            scenarios.append(scenario)
        
        elif 'validate' in func_name or 'check' in func_name or 'verify' in func_name:
            scenario = {
                'name': f"{name}",
                'given': "Input data needs validation",
                'when': f"The function {func_name} is called with data to validate",
                'then': "The data is correctly validated"
            }
            scenarios.append(scenario)
        
        else:
            # Generic scenario for other functions
            scenario = {
                'name': f"{name}",
                'given': "The system is in a valid state",
                'when': f"The function {func_name} is called",
                'then': "The expected operation is performed successfully"
            }
            scenarios.append(scenario)
    
    return scenarios

def generate_test_cases_from_scenarios(scenarios: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Generate clear, understandable test cases from BDD scenarios.

    Args:
        scenarios: List of dictionaries with Given-When-Then scenarios

    Returns:
        list: List of test case dictionaries
    """
    test_cases = []
    
    for i, scenario in enumerate(scenarios):
        name = scenario.get('name', f"Test scenario {i+1}")
        given = scenario.get('given', "")
        when = scenario.get('when', "")
        then = scenario.get('then', "")
        
        # Generate steps
        steps = _create_simplified_steps(given, when, then)
        
        test_case = {
            'name': f"BDD: {name}",
            'description': f"Behavior-driven test to verify: {name}",
            'preconditions': given,
            'expected_result': then,
            'steps': steps
        }
        
        test_cases.append(test_case)
    
    return test_cases

def _create_simplified_steps(given: str, when: str, then: str) -> List[str]:
    """Create simplified test steps from Given-When-Then statements."""
    steps = []
    
    # Break down the Given condition into setup steps
    given_steps = given.split(" and ")
    for step in given_steps:
        step = step.strip()
        if step:
            steps.append(f"Set up: {step}")
    
    # Add the When action as a step
    when_steps = when.split(" and ")
    for step in when_steps:
        step = step.strip()
        if step:
            steps.append(f"Action: {step}")
    
    # Add the Then verification as a step
    then_steps = then.split(" and ")
    for step in then_steps:
        step = step.strip()
        if step:
            steps.append(f"Verify: {step}")
    
    return steps

def _create_negative_scenario(scenario: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Create a negative scenario based on a positive one."""
    name = scenario.get('name', "")
    given = scenario.get('given', "")
    when = scenario.get('when', "")
    then = scenario.get('then', "")
    
    # Skip if already a negative scenario
    if any(term in name.lower() for term in ['invalid', 'error', 'fail', 'negative', 'exception']):
        return None
    
    # Create negative version
    negative_scenario = {
        'name': f"{name} with invalid input",
        'given': given,
        'when': when.replace("valid", "invalid"),
        'then': "An appropriate error is raised or operation is gracefully rejected"
    }
    
    return negative_scenario

def generate_framework_specific_tests(test_cases: List[Dict[str, Any]], framework: str = "pytest") -> str:
    """
    Generate framework-specific test code from test cases.

    Args:
        test_cases: List of test case dictionaries
        framework: Test framework to generate code for (pytest, unittest, jest, etc.)

    Returns:
        str: Generated test code
    """
    code = ""
    
    if framework == "pytest":
        code = "import pytest\n\n"
        
        for i, tc in enumerate(test_cases):
            func_name = f"test_{tc['name'].lower().replace(' ', '_').replace(':', '').replace('-', '_')}"
            code += f"def {func_name}():\n"
            code += f'    """{tc["description"]}\n\n'
            code += f"    Preconditions: {tc['preconditions']}\n"
            code += f"    Expected Result: {tc['expected_result']}\n"
            code += '    """\n'
            
            for step in tc['steps']:
                if step.startswith("Set up:"):
                    code += f"    # {step}\n"
                elif step.startswith("Action:"):
                    code += f"    # {step}\n"
                elif step.startswith("Verify:"):
                    code += f"    # {step}\n"
                    code += f"    assert True  # Replace with actual verification\n"
            
            code += "\n"
    
    elif framework == "unittest":
        code = "import unittest\n\n"
        code += "class BehaviorDrivenTests(unittest.TestCase):\n\n"
        
        for i, tc in enumerate(test_cases):
            func_name = f"test_{tc['name'].lower().replace(' ', '_').replace(':', '').replace('-', '_')}"
            code += f"    def {func_name}(self):\n"
            code += f'        """{tc["description"]}\n\n'
            code += f"        Preconditions: {tc['preconditions']}\n"
            code += f"        Expected Result: {tc['expected_result']}\n"
            code += '        """\n'
            
            for step in tc['steps']:
                if step.startswith("Set up:"):
                    code += f"        # {step}\n"
                elif step.startswith("Action:"):
                    code += f"        # {step}\n"
                elif step.startswith("Verify:"):
                    code += f"        # {step}\n"
                    code += f"        self.assertTrue(True)  # Replace with actual verification\n"
            
            code += "\n"
        
        code += "if __name__ == '__main__':\n"
        code += "    unittest.main()\n"
    
    else:
        # Default to simple text format
        for i, tc in enumerate(test_cases):
            code += f"Test: {tc['name']}\n"
            code += f"Description: {tc['description']}\n"
            code += f"Preconditions: {tc['preconditions']}\n"
            code += "Steps:\n"
            
            for step in tc['steps']:
                code += f"- {step}\n"
            
            code += f"Expected Result: {tc['expected_result']}\n\n"
    
    return code

def generate_behavior_driven_tests(source_type: str, source_content: str) -> List[Dict[str, Any]]:
    """
    Generate behavior-driven tests from source code or requirements.
    
    Args:
        source_type: 'code' or 'requirements'
        source_content: The source code or requirements text
        
    Returns:
        List of test case dictionaries
    """
    if source_type == 'code':
        scenarios = extract_bdd_scenarios_from_code(source_content)
    else:  # source_type == 'requirements'
        scenarios = extract_bdd_scenarios(source_content)
    
    test_cases = generate_test_cases_from_scenarios(scenarios)
    return test_cases