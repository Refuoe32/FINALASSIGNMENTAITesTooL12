"""
Mutation Testing Implementation
Generates tests that change code to verify that tests can detect these changes
"""
import re
from typing import List, Dict, Any

def generate_mutation_tests(source_type: str, source_content: str) -> List[Dict[str, Any]]:
    """
    Generate mutation tests from source code or requirements

    Args:
        source_type: 'code' or 'requirements'
        source_content: The source code or requirements text

    Returns:
        List of test cases with name, description, and test code
    """
    if source_type == 'code':
        return _generate_from_code(source_content)
    else:  # source_type == 'requirements'
        return _generate_from_requirements(source_content)

def _generate_from_code(code: str) -> List[Dict[str, Any]]:
    """Generate mutation tests from source code"""
    test_cases = []
    
    # Extract functions to test
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
    functions = re.findall(function_pattern, code)
    
    # Look for conditions that can be mutated
    condition_patterns = [
        (r'if\s+(.*?):', 'conditional'),
        (r'while\s+(.*?):', 'loop condition'),
        (r'return\s+(.*?)(?:\s|$)', 'return value'),
        (r'(\w+)\s*=\s*(.*?)(?:\s|$)', 'assignment')
    ]
    
    all_funcs = [f[0] for f in functions if not f[0].startswith('_')]
    
    # If no functions found, create generic test cases
    if not all_funcs:
        test_cases.append({
            'name': "Boundary condition mutation",
            'description': "Verify that boundary condition checks are implemented correctly",
            'preconditions': "Source code is available",
            'expected_result': "Mutated boundary conditions should be detected by tests",
            'steps': [
                "Identify boundary conditions in the code",
                "Create mutations that change boundary conditions (< to <=, > to >=, etc.)",
                "Verify that tests detect these mutations"
            ]
        })
        
        test_cases.append({
            'name': "Return value mutation",
            'description': "Verify that return value checks are implemented correctly",
            'preconditions': "Source code is available",
            'expected_result': "Mutated return values should be detected by tests",
            'steps': [
                "Identify return statements in the code",
                "Create mutations that change return values (return true to false, return null instead of value, etc.)",
                "Verify that tests detect these mutations"
            ]
        })
        
        test_cases.append({
            'name': "Arithmetic operator mutation",
            'description': "Verify that arithmetic operations are implemented correctly",
            'preconditions': "Source code is available",
            'expected_result': "Mutated arithmetic operations should be detected by tests",
            'steps': [
                "Identify arithmetic operations in the code",
                "Create mutations that change operators (+ to -, * to /, etc.)",
                "Verify that tests detect these mutations"
            ]
        })
        
        return test_cases
    
    # Create test cases for specific functions and conditions
    for func_name in all_funcs[:3]:  # Limit to 3 functions
        # Find the function body (simplified approach)
        func_def_start = code.find(f"def {func_name}")
        if func_def_start == -1:
            continue
            
        # Find the end of the function (approximate)
        next_def = code.find("def ", func_def_start + 1)
        if next_def != -1:
            func_body = code[func_def_start:next_def]
        else:
            func_body = code[func_def_start:]
        
        # Look for conditions to mutate
        for pattern, cond_type in condition_patterns:
            conditions = re.findall(pattern, func_body)
            
            for i, condition in enumerate(conditions[:2]):  # Limit to 2 conditions per type
                if isinstance(condition, tuple):
                    condition = condition[0]  # Extract first group if multiple groups
                
                # Skip empty or very short conditions
                if not condition or len(condition.strip()) < 3:
                    continue
                
                # Generate mutation strategies based on condition type
                if cond_type == 'conditional' or cond_type == 'loop condition':
                    test_case = {
                        'name': f"Mutate {cond_type} in {func_name}",
                        'description': f"Test that verifies the {cond_type} '{condition}' in function {func_name} is correctly implemented",
                        'preconditions': "Source code is available and function is accessible",
                        'expected_result': "Tests should detect when the condition is inverted or modified",
                        'steps': [
                            f"Identify the {cond_type} '{condition}' in function {func_name}",
                            f"Create a mutation that inverts the condition",
                            "Run tests to verify they detect the mutation",
                            f"Create a mutation that replaces the condition with 'True'",
                            "Run tests to verify they detect the mutation",
                            f"Create a mutation that replaces the condition with 'False'",
                            "Run tests to verify they detect the mutation"
                        ]
                    }
                    test_cases.append(test_case)
                
                elif cond_type == 'return value':
                    test_case = {
                        'name': f"Mutate return value in {func_name}",
                        'description': f"Test that verifies the return value '{condition}' in function {func_name} is correctly validated",
                        'preconditions': "Source code is available and function is accessible",
                        'expected_result': "Tests should detect when the return value is modified",
                        'steps': [
                            f"Identify the return statement '{condition}' in function {func_name}",
                            f"Create a mutation that changes the return value",
                            "Run tests to verify they detect the mutation",
                            f"Create a mutation that returns None/null instead",
                            "Run tests to verify they detect the mutation"
                        ]
                    }
                    test_cases.append(test_case)
                
                elif cond_type == 'assignment':
                    test_case = {
                        'name': f"Mutate assignment in {func_name}",
                        'description': f"Test that verifies the assignment '{condition}' in function {func_name} is correctly implemented",
                        'preconditions': "Source code is available and function is accessible",
                        'expected_result': "Tests should detect when the assignment is modified",
                        'steps': [
                            f"Identify the assignment '{condition}' in function {func_name}",
                            f"Create a mutation that changes the assigned value",
                            "Run tests to verify they detect the mutation"
                        ]
                    }
                    test_cases.append(test_case)
        
        # If no specific conditions found, create generic function-level test cases
        if not test_cases:
            test_case = {
                'name': f"Boundary condition mutation for {func_name}",
                'description': f"Verify that boundary checks in {func_name} are implemented correctly",
                'preconditions': "Source code is available and function is accessible",
                'expected_result': "Mutations to boundary conditions should be detected by tests",
                'steps': [
                    f"Identify boundary conditions in the function {func_name}",
                    "Create mutations that change boundary conditions (< to <=, > to >=, etc.)",
                    "Verify that tests detect these mutations"
                ]
            }
            test_cases.append(test_case)
    
    # Add generic operator mutation tests
    test_cases.append({
        'name': "Arithmetic operator mutation",
        'description': "Test that verifies arithmetic operations are correctly implemented",
        'preconditions': "Source code is available",
        'expected_result': "Mutations to arithmetic operators should be detected by tests",
        'steps': [
            "Identify arithmetic operations in the code",
            "Create mutations that change operators (+ to -, * to /, etc.)",
            "Verify that tests detect these mutations"
        ]
    })
    
    test_cases.append({
        'name': "Logical operator mutation",
        'description': "Test that verifies logical operations are correctly implemented",
        'preconditions': "Source code is available",
        'expected_result': "Mutations to logical operators should be detected by tests",
        'steps': [
            "Identify logical operations in the code",
            "Create mutations that change operators (AND to OR, OR to AND, etc.)",
            "Verify that tests detect these mutations"
        ]
    })
    
    return test_cases

def _generate_from_requirements(requirements: str) -> List[Dict[str, Any]]:
    """Generate mutation tests from requirements text"""
    test_cases = []
    
    # Extract key business rules that can be mutated
    rule_patterns = [
        r'(?:must|should|shall|will|needs to)\s+([^.]*)',
        r'(?:is|are) required to\s+([^.]*)',
        r'(?:only|always|never)\s+([^.]*)'
    ]
    
    all_rules = []
    for pattern in rule_patterns:
        rules = re.findall(pattern, requirements, re.IGNORECASE)
        all_rules.extend([r.strip() for r in rules if len(r.strip()) > 10])
    
    # If no specific rules found, create generic test cases
    if not all_rules:
        # Business rule mutations
        test_cases.append({
            'name': "Business rule boundary mutation",
            'description': "Verify that boundary conditions in business rules are correctly implemented",
            'preconditions': "Application implements the specified business rules",
            'expected_result': "Mutations to business rule boundaries should cause tests to fail",
            'steps': [
                "Identify boundary values in business rules",
                "Create test cases that mutate these boundaries (e.g., minimum value - 1, maximum value + 1)",
                "Verify that the system correctly enforces the original boundaries"
            ]
        })
        
        # Validation rule mutations
        test_cases.append({
            'name': "Data validation rule mutation",
            'description': "Verify that data validation rules are correctly implemented",
            'preconditions': "Application implements data validation rules",
            'expected_result': "Mutations to validation rules should cause tests to fail",
            'steps': [
                "Identify data validation rules",
                "Create test cases that bypass these validations",
                "Verify that the system correctly enforces the original validation rules"
            ]
        })
        
        # Workflow mutations
        test_cases.append({
            'name': "Workflow sequence mutation",
            'description': "Verify that workflow sequences are correctly enforced",
            'preconditions': "Application implements specific workflow sequences",
            'expected_result': "Mutations to workflow sequences should cause tests to fail",
            'steps': [
                "Identify required sequences in workflows",
                "Create test cases that attempt to perform steps out of sequence",
                "Verify that the system correctly enforces the original sequence"
            ]
        })
        
        return test_cases
    
    # Create test cases for specific rules
    for i, rule in enumerate(all_rules[:5]):  # Limit to 5 rules
        # Determine rule type for better mutation strategies
        rule_type = "general"
        
        if any(term in rule.lower() for term in ['equal', 'greater', 'less', 'minimum', 'maximum', 'at least', 'at most']):
            rule_type = "boundary"
        elif any(term in rule.lower() for term in ['valid', 'format', 'pattern', 'match']):
            rule_type = "validation"
        elif any(term in rule.lower() for term in ['before', 'after', 'following', 'sequence', 'workflow']):
            rule_type = "sequence"
        elif any(term in rule.lower() for term in ['only', 'exclusively', 'restricted']):
            rule_type = "restriction"
        
        # Create appropriate test case based on rule type
        if rule_type == "boundary":
            test_case = {
                'name': f"Boundary rule mutation",
                'description': f"Test that verifies the boundary rule '{rule}' is correctly implemented",
                'preconditions': "Application implements the specified boundary rule",
                'expected_result': "System should enforce the correct boundary condition",
                'steps': [
                    f"Identify the boundary rule: '{rule}'",
                    "Create test cases for the boundary value",
                    "Create test cases for boundary value - 1",
                    "Create test cases for boundary value + 1",
                    "Verify that the system enforces the original boundary rule"
                ]
            }
        
        elif rule_type == "validation":
            test_case = {
                'name': f"Validation rule mutation",
                'description': f"Test that verifies the validation rule '{rule}' is correctly implemented",
                'preconditions': "Application implements the specified validation rule",
                'expected_result': "System should enforce the correct validation",
                'steps': [
                    f"Identify the validation rule: '{rule}'",
                    "Create test cases with valid data according to the rule",
                    "Create test cases with invalid data that violates the rule",
                    "Create test cases with edge cases for the validation",
                    "Verify that the system enforces the original validation rule"
                ]
            }
        
        elif rule_type == "sequence":
            test_case = {
                'name': f"Sequence rule mutation",
                'description': f"Test that verifies the sequence rule '{rule}' is correctly implemented",
                'preconditions': "Application implements the specified sequence rule",
                'expected_result': "System should enforce the correct sequence",
                'steps': [
                    f"Identify the sequence rule: '{rule}'",
                    "Create test cases that follow the correct sequence",
                    "Create test cases that attempt steps out of sequence",
                    "Create test cases that skip steps in the sequence",
                    "Verify that the system enforces the original sequence rule"
                ]
            }
        
        elif rule_type == "restriction":
            test_case = {
                'name': f"Restriction rule mutation",
                'description': f"Test that verifies the restriction rule '{rule}' is correctly implemented",
                'preconditions': "Application implements the specified restriction rule",
                'expected_result': "System should enforce the correct restrictions",
                'steps': [
                    f"Identify the restriction rule: '{rule}'",
                    "Create test cases that comply with the restriction",
                    "Create test cases that violate the restriction",
                    "Verify that the system enforces the original restriction rule"
                ]
            }
        
        else:  # general rule
            test_case = {
                'name': f"Business rule mutation",
                'description': f"Test that verifies the business rule '{rule}' is correctly implemented",
                'preconditions': "Application implements the specified business rule",
                'expected_result': "System should correctly implement the business rule",
                'steps': [
                    f"Identify the business rule: '{rule}'",
                    "Create test cases that follow the rule",
                    "Create test cases that violate the rule",
                    "Verify that the system enforces the original business rule"
                ]
            }
        
        test_cases.append(test_case)
    
    return test_cases