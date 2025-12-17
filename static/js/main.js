// California House Prediction - Main JavaScript
// Handles UI interactions, API calls, and animations

// Global loading state management
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    const text = overlay.querySelector('.loading-text');
    text.textContent = message;
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.remove('active');
}

// Toast notifications
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Navigation menu toggle for mobile
const navToggle = document.getElementById('navToggle');
const navMenu = document.getElementById('navMenu');

if (navToggle && navMenu) {
    // Toggle menu on button click
    navToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        navMenu.classList.toggle('active');
        navToggle.classList.toggle('active');
    });
    
    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!navMenu.contains(e.target) && !navToggle.contains(e.target)) {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        }
    });
    
    // Close menu when clicking a nav link
    const navLinks = navMenu.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        });
    });
    
    // Close menu on window resize if desktop size
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            navMenu.classList.remove('active');
            navToggle.classList.remove('active');
        }
    });
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Form validation helpers
function validateNumber(value, min = -Infinity, max = Infinity) {
    const num = parseFloat(value);
    return !isNaN(num) && num >= min && num <= max;
}

function highlightError(element) {
    element.style.borderColor = 'var(--error)';
    setTimeout(() => {
        element.style.borderColor = '';
    }, 2000);
}

// Batch prediction helpers
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Number formatting
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

// API helper function
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Debounce function for input validation
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add animation class when element enters viewport
function observeElements() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, {
        threshold: 0.1
    });
    
    document.querySelectorAll('.feature-card, .stat-card, .step').forEach(el => {
        observer.observe(el);
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    observeElements();
    
    // Add enter key listener to forms
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    });
});

// Export functions for use in other scripts
window.AppUtils = {
    showLoading,
    hideLoading,
    showToast,
    apiCall,
    formatCurrency,
    formatFileSize,
    validateNumber,
    highlightError
};
