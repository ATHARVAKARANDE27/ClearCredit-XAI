/* --- ClearCredit XAI | Core Interaction Layer --- */

document.addEventListener('DOMContentLoaded', () => {
    initStars();
    initWizard();
    initFormFormatting();
});

/* --- Star Field Generator --- */
function initStars() {
    const field = document.getElementById('starField');
    if (!field) return;

    const count = 100;
    for (let i = 0; i < count; i++) {
        const star = document.createElement('div');
        const size = Math.random() * 3 + 1;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const duration = Math.random() * 3 + 2;

        star.className = 'star';
        star.style.width = `${size}px`;
        star.style.height = `${size}px`;
        star.style.left = `${x}%`;
        star.style.top = `${y}%`;
        star.style.setProperty('--duration', `${duration}s`);

        field.appendChild(star);
    }
}

/* --- Multi-step Wizard Logic --- */
function initWizard() {
    const form = document.getElementById('riskForm');
    if (!form) return;

    const steps = Array.from(document.querySelectorAll('.form-step'));
    const nextBtn = document.getElementById('nextBtn');
    const prevBtn = document.getElementById('prevBtn');
    const progressFill = document.getElementById('progressFill');
    let currentStep = 0;

    const updateUI = () => {
        steps.forEach((step, idx) => {
            step.classList.toggle('active', idx === currentStep);
        });

        // Update Nav
        prevBtn.style.visibility = currentStep === 0 ? 'hidden' : 'visible';

        if (currentStep === steps.length - 1) {
            nextBtn.style.display = 'none';
        } else {
            nextBtn.style.display = 'flex';
            nextBtn.innerText = 'Next →';
        }

        // Update Progress
        const percent = ((currentStep + 1) / steps.length) * 100;
        progressFill.style.width = `${percent}%`;

        // Auto-calculate LTI if on last step
        if (currentStep === 2) {
            calculateLTI();
        }
    };

    nextBtn.addEventListener('click', () => {
        if (validateStep(currentStep)) {
            currentStep++;
            updateUI();
        }
    });

    prevBtn.addEventListener('click', () => {
        currentStep--;
        updateUI();
    });

    form.addEventListener('submit', () => {
        document.getElementById('loaderOverlay').style.display = 'flex';
    });

    updateUI();
}

/* --- Validation --- */
function validateStep(stepIdx) {
    const step = document.querySelectorAll('.form-step')[stepIdx];
    const inputs = step.querySelectorAll('input[required]');
    let valid = true;

    inputs.forEach(input => {
        if (!input.value) {
            valid = false;
            input.closest('.input-group')?.classList.add('error');
            // Remove error class on input
            input.addEventListener('input', () => {
                input.closest('.input-group')?.classList.remove('error');
            }, { once: true });
        }
    });

    return valid;
}

/* --- Selection Cards --- */
window.selectCard = (element, inputId) => {
    const value = element.getAttribute('data-value');
    const input = document.getElementById(inputId);
    input.value = value;

    // UI Updates
    const grid = element.parentElement;
    grid.querySelectorAll('.selection-card').forEach(card => card.classList.remove('selected'));
    element.classList.add('selected');
};

/* --- Form Formatting --- */
function initFormFormatting() {
    const numberInputs = document.querySelectorAll('.number-format');

    numberInputs.forEach(input => {
        input.addEventListener('input', (e) => {
            let val = e.target.value.replace(/[^0-9]/g, '');
            if (val) {
                e.target.value = Number(val).toLocaleString();
            }
        });
    });
}

function calculateLTI() {
    const income = Number(document.getElementById('income').value.replace(/[^0-9]/g, ''));
    const loan = Number(document.getElementById('amnt').value.replace(/[^0-9]/g, ''));

    if (income && loan) {
        const lti = (loan / income).toFixed(2);
        document.getElementById('lti').value = lti;
    }
}
