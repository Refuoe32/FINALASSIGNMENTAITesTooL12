"""
Fuzz Testing Implementation
Generates tests using random inputs to find bugs
"""
import re
import random
from typing import List, Dict, Any

def generate_fuzz_tests(source_type: str, source_content: str) -> List[Dict[str, Any]]:
    """
    Generate fuzz tests from source code or requirements

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
    """Generate fuzz tests from source code"""
    test_cases = []
    
    # Extract functions to test
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)'
    functions = re.findall(function_pattern, code)
    
    # For each function found, create fuzz test cases
    for func_name, params in functions:
        # Skip private methods
        if func_name.startswith('_'):
            continue
            
        # Parse parameters
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
        
        # Skip functions without parameters - not good candidates for fuzzing
        if not param_list:
            continue
        
        # Create random input test
        test_cases.append({
            'name': f"Fuzz test: {func_name} with random inputs",
            'description': f"Test {func_name} with completely random inputs to detect crashes or unexpected behavior",
            'preconditions': f"Function {func_name} is accessible",
            'expected_result': "Function handles unexpected inputs gracefully without crashing",
            'steps': [
                f"Prepare random inputs for each parameter of {func_name}",
                "Execute function with random inputs",
                "Verify the function handles unexpected inputs gracefully"
            ]
        })
        
        # Create boundary value test
        test_cases.append({
            'name': f"Fuzz test: {func_name} with boundary values",
            'description': f"Test {func_name} with boundary values and edge cases to detect incorrect handling",
            'preconditions': f"Function {func_name} is accessible",
            'expected_result': "Function handles boundary values correctly",
            'steps': [
                f"Prepare boundary values for each parameter of {func_name} (empty strings, zero, negative numbers, extremely large values, etc.)",
                "Execute function with boundary values",
                "Verify the function handles boundary values correctly"
            ]
        })
        
        # Create format string vulnerability test
        test_cases.append({
            'name': f"Fuzz test: {func_name} with format string attacks",
            'description': f"Test {func_name} with format string patterns to detect format string vulnerabilities",
            'preconditions': f"Function {func_name} is accessible",
            'expected_result': "Function is not vulnerable to format string attacks",
            'steps': [
                f"Prepare format string patterns for string parameters of {func_name} (%s, %d, %x, etc.)",
                "Execute function with format string patterns",
                "Verify the function is not vulnerable to format string attacks"
            ]
        })
    
    # If no functions found, create generic fuzz tests
    if not test_cases:
        test_cases.append({
            'name': "Fuzz test: Random input fields",
            'description': "Test application with random inputs in all input fields",
            'preconditions': "Application is available for testing",
            'expected_result': "Application handles random inputs gracefully without crashing",
            'steps': [
                "Identify all input fields in the application",
                "Generate random inputs for each field (strings, numbers, special characters, etc.)",
                "Submit forms with random inputs",
                "Verify the application handles random inputs gracefully"
            ]
        })
        
        test_cases.append({
            'name': "Fuzz test: Boundary values",
            'description': "Test application with boundary values in all input fields",
            'preconditions': "Application is available for testing",
            'expected_result': "Application handles boundary values correctly",
            'steps': [
                "Identify all input fields in the application",
                "Generate boundary values for each field (empty values, maximum length, zero, negative numbers, etc.)",
                "Submit forms with boundary values",
                "Verify the application handles boundary values correctly"
            ]
        })
        
        test_cases.append({
            'name': "Fuzz test: Format string attacks",
            'description': "Test application with format string patterns to detect vulnerabilities",
            'preconditions': "Application is available for testing",
            'expected_result': "Application is not vulnerable to format string attacks",
            'steps': [
                "Identify all text input fields in the application",
                "Generate format string patterns (%s, %d, %x, etc.)",
                "Submit forms with format string patterns",
                "Verify the application is not vulnerable to format string attacks"
            ]
        })
    
    return test_cases

def _generate_from_requirements(requirements: str) -> List[Dict[str, Any]]:
    """Generate fuzz tests from requirements text"""
    test_cases = []
    
    # Extract potential input fields
    input_fields = _extract_input_fields(requirements)
    
    # Generate test cases for each input field
    for field in input_fields:
        field_name = field.get('name', 'input field')
        
        # Random input test
        test_cases.append({
            'name': f"Fuzz test: {field_name} with random inputs",
            'description': f"Test {field_name} with completely random inputs to detect crashes or unexpected behavior",
            'preconditions': "System under test is available",
            'expected_result': f"System handles unexpected inputs for {field_name} gracefully",
            'steps': [
                f"Prepare random inputs for {field_name}",
                f"Submit {field_name} with random inputs",
                "Verify the system handles unexpected inputs gracefully"
            ]
        })
        
        # Boundary value test
        test_cases.append({
            'name': f"Fuzz test: {field_name} with boundary values",
            'description': f"Test {field_name} with boundary values and edge cases to detect incorrect handling",
            'preconditions': "System under test is available",
            'expected_result': f"System handles boundary values for {field_name} correctly",
            'steps': [
                f"Prepare boundary values for {field_name} (empty values, maximum length, zero, negative numbers, etc.)",
                f"Submit {field_name} with boundary values",
                "Verify the system handles boundary values correctly"
            ]
        })
    
    # If no specific input fields identified, create generic test cases
    if not test_cases:
        # Random input test for entire application
        test_cases.append({
            'name': "Fuzz test: Application with random inputs",
            'description': "Test application with random inputs to detect crashes or unexpected behavior",
            'preconditions': "Application is available for testing",
            'expected_result': "Application handles unexpected inputs gracefully",
            'steps': [
                "Identify all input fields in the application",
                "Generate random inputs for each field",
                "Submit forms with random inputs",
                "Verify the application handles unexpected inputs gracefully"
            ]
        })
        
        # API fuzz test
        test_cases.append({
            'name': "Fuzz test: API endpoints",
            'description': "Test API endpoints with unexpected requests to detect vulnerabilities",
            'preconditions': "API is available for testing",
            'expected_result': "API handles unexpected requests gracefully",
            'steps': [
                "Identify all API endpoints",
                "Generate requests with unexpected parameters and values",
                "Send requests to API endpoints",
                "Verify the API handles unexpected requests gracefully"
            ]
        })
        
        # Workflow fuzz test
        test_cases.append({
            'name': "Fuzz test: Application workflows",
            'description': "Test application workflows with unexpected sequences to detect vulnerabilities",
            'preconditions': "Application is available for testing",
            'expected_result': "Application handles unexpected sequences gracefully",
            'steps': [
                "Identify all application workflows",
                "Generate unexpected sequences of actions",
                "Execute unexpected sequences",
                "Verify the application handles unexpected sequences gracefully"
            ]
        })
    
    return test_cases

def _extract_input_fields(requirements: str) -> List[Dict[str, Any]]:
    """Extract potential input fields from requirements"""
    input_fields = []
    
    # Look for input field patterns
    input_patterns = [
        r'(?:input|field|form|text|select|dropdown|checkbox|radio|button)\s+(?:for|called|named|labeled)?\s*[\'"]?([a-zA-Z0-9_\s]+)[\'"]?',
        r'(?:enter|input|provide|specify|select)\s+(?:the|a|an)?\s*([a-zA-Z0-9_\s]+)',
        r'(?:the|a|an)\s+([a-zA-Z0-9_\s]+)\s+(?:field|input|form|text|select|dropdown|checkbox|radio|button)'
    ]
    
    for pattern in input_patterns:
        matches = re.findall(pattern, requirements, re.IGNORECASE)
        for match in matches:
            field_name = match.strip()
            if len(field_name) > 2 and not any(field['name'] == field_name for field in input_fields):
                # Try to determine field type
                field_type = 'text'  # Default type
                if any(term in requirements.lower() for term in [f"{field_name} number", f"numeric {field_name}", f"{field_name} int"]):
                    field_type = 'number'
                elif any(term in requirements.lower() for term in [f"{field_name} date", f"{field_name} time"]):
                    field_type = 'date'
                elif any(term in requirements.lower() for term in [f"{field_name} email"]):
                    field_type = 'email'
                elif any(term in requirements.lower() for term in [f"{field_name} checkbox", f"{field_name} boolean"]):
                    field_type = 'boolean'
                
                input_fields.append({
                    'name': field_name,
                    'type': field_type
                })
    
    # If no input fields found, create some common ones
    if not input_fields:
        common_fields = [
            {'name': 'username', 'type': 'text'},
            {'name': 'password', 'type': 'text'},
            {'name': 'email', 'type': 'email'},
            {'name': 'name', 'type': 'text'},
            {'name': 'address', 'type': 'text'},
            {'name': 'phone', 'type': 'text'},
            {'name': 'date', 'type': 'date'},
            {'name': 'quantity', 'type': 'number'}
        ]
        
        for field in common_fields:
            if field['name'].lower() in requirements.lower():
                input_fields.append(field)
        
        # If still no fields found, add some generic ones
        if not input_fields:
            input_fields = [
                {'name': 'text input', 'type': 'text'},
                {'name': 'numeric input', 'type': 'number'},
                {'name': 'selection input', 'type': 'select'}
            ]
    
    return input_fields