/**
 * Main JavaScript file for AI Test Tool
 */

document.addEventListener('DOMContentLoaded', function() {
    // Apply theme from localStorage if available
    applyThemeFromStorage();
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Handle active navigation
    setActiveNavItem();
});

/**
 * Apply saved theme settings from localStorage
 */
function applyThemeFromStorage() {
    // Apply theme
    if (localStorage.getItem('appTheme')) {
        document.documentElement.setAttribute('data-bs-theme', localStorage.getItem('appTheme'));
    }
    
    // Apply background color
    if (localStorage.getItem('backgroundColor')) {
        document.body.style.backgroundColor = localStorage.getItem('backgroundColor');
    }
    
    // Apply text color
    if (localStorage.getItem('textColor')) {
        document.body.style.color = localStorage.getItem('textColor');
    }
}

/**
 * Set active navigation item based on current page
 */
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    
    // For main nav items in top navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        } else if (currentPath.includes('/results/') && link.getAttribute('href') === '/history') {
            // Handle results page being part of history
            link.classList.add('active');
        }
    });
    
    // For sidebar links
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    sidebarLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && href !== '#' && currentPath === href) {
            link.classList.add('active');
        } else if (href && href !== '#' && currentPath.includes('/results/') && href === '/history') {
            // Handle results page being part of history
            link.classList.add('active');
        }
    });
}

/**
 * Format test ID with proper formatting
 * @param {number} id - Numeric ID to format
 * @returns {string} Formatted ID in TC-X format
 */
function formatTestId(id) {
    return `TC-${id}`;
}

/**
 * Delete a test case and update UI
 * @param {number} testCaseId - ID of the test case to delete
 * @param {Element} element - DOM element to remove on success
 */
function deleteTestCase(testCaseId, element) {
    if (confirm('Are you sure you want to delete this test case?')) {
        fetch(`/api/testcases/${testCaseId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Remove the element from DOM
                const container = element.closest('.accordion-item, tr');
                if (container) {
                    container.remove();
                }
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while deleting the test case');
        });
    }
}

/**
 * Delete a test run and update UI
 * @param {number} runId - ID of the test run to delete
 * @param {Element} element - DOM element to remove on success
 */
function deleteTestRun(runId, element) {
    if (confirm('Are you sure you want to delete this test run?')) {
        fetch(`/api/runs/${runId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Remove the element from DOM
                const container = element.closest('tr, .card');
                if (container) {
                    container.remove();
                }
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while deleting the test run');
        });
    }
}

/**
 * Delete a project and update UI
 * @param {number} projectId - ID of the project to delete
 * @param {Element} element - DOM element to remove on success
 */
function deleteProject(projectId, element) {
    if (confirm('Are you sure you want to delete this project and all associated test runs?')) {
        fetch(`/api/projects/${projectId}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Remove the element from DOM
                const container = element.closest('tr, .card');
                if (container) {
                    container.remove();
                }
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while deleting the project');
        });
    }
}