// Global function for Card Selection
window.selectCard = (card, inputId) => {
    // Update Hidden Input
    const input = document.getElementById(inputId);
    if (input) {
        input.value = card.dataset.value;
        // Trigger generic change event for validation if needed
        input.dispatchEvent(new Event('change'));
    }

    // Update Visuals
    const siblings = card.parentElement.querySelectorAll('.selection-card');
    siblings.forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
};

document.addEventListener('DOMContentLoaded', () => {

    // --- Formatting Utils ---
    const formatCurrency = (num) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            maximumFractionDigits: 0
        }).format(num);
    };

    // --- Number Input Comma Formatting (for text inputs) ---
    // Moved up for better organization
    const numberInputs = document.querySelectorAll('.number-format');
    numberInputs.forEach(input => {
        input.addEventListener('input', (e) => {
            // Remove non-digits
            let val = e.target.value.replace(/\D/g, '');
            if (val) {
                // Add commas
                e.target.value = parseInt(val).toLocaleString('en-US');
            }
        });
    });

    // --- Auto-calculate LTI Ratio ---
    const incomeInput = document.getElementById('income');
    const amountInput = document.getElementById('amnt');
    const ltiInput = document.getElementById('lti');

    const calculateLTI = () => {
        const income = parseFloat(incomeInput.value.replace(/[^0-9.]/g, '')) || 0;
        const amount = parseFloat(amountInput.value.replace(/[^0-9.]/g, '')) || 0;

        if (income > 0) {
            const ratio = (amount / income).toFixed(2);
            ltiInput.value = Math.min(ratio, 1.0); // Clamp to 1.0 for the model's standard
        } else {
            ltiInput.value = "";
        }
    };

    if (incomeInput && amountInput && ltiInput) {
        incomeInput.addEventListener('input', calculateLTI);
        amountInput.addEventListener('input', calculateLTI);
    }

    const initWizard = () => {
        const steps = document.querySelectorAll('.form-step');
        const stepItems = document.querySelectorAll('.step-item'); // Changed selector
        const progressFill = document.querySelector('.progress-fill'); // New selector
        const nextBtn = document.getElementById('nextBtn');
        const prevBtn = document.getElementById('prevBtn');
        const submitBtn = document.getElementById('submitBtn');
        let currentStep = 0;

        if (!steps.length) return;

        const updateWizard = () => {
            // Update Steps Visibility
            steps.forEach((step, index) => {
                if (index === currentStep) {
                    step.classList.add('active');
                } else {
                    step.classList.remove('active');
                }
            });

            // Update Indicators & Progress Bar
            const progress = (currentStep / (steps.length - 1)) * 100;
            if (progressFill) {
                progressFill.style.width = `${progress}%`;
            }

            stepItems.forEach((item, index) => {
                const circle = item.querySelector('.step-circle');

                if (index < currentStep) {
                    item.classList.add('completed');
                    item.classList.remove('active');
                    circle.innerHTML = '<i class="fas fa-check"></i>';
                } else if (index === currentStep) {
                    item.classList.add('active');
                    item.classList.remove('completed');
                    circle.innerHTML = index + 1;
                } else {
                    item.classList.remove('active', 'completed');
                    circle.innerHTML = index + 1;
                }
            });

            // Buttons
            prevBtn.style.display = currentStep === 0 ? 'none' : 'block';
            if (currentStep === steps.length - 1) {
                nextBtn.style.display = 'none';
                submitBtn.parentElement.style.display = 'block'; // Show submit container
            } else {
                nextBtn.style.display = 'block';
                submitBtn.parentElement.style.display = 'none';
            }
        };

        nextBtn.addEventListener('click', () => {
            // Validate current step inputs (including hidden ones)
            const currentInputs = steps[currentStep].querySelectorAll('input, select');
            let isValid = true;

            // Check HTML5 validity
            currentInputs.forEach(input => {
                if (!input.checkValidity()) {
                    isValid = false;
                    input.reportValidity();
                }
            });

            // Extra check for hidden inputs (card selection)
            // reportValidity() doesn't always work on hidden inputs, so we might need a toast/alert
            // But if 'required' is set on hidden input, checkValidity might return false.
            // Let's manually check hidden required inputs if checkValidity didn't trigger UI
            if (isValid) {
                const hiddenRequired = steps[currentStep].querySelectorAll('input[type="hidden"][required]');
                hiddenRequired.forEach(input => {
                    if (!input.value) {
                        isValid = false;
                        // Shake animation or alert?
                        // For now, simpler: Scroll to it or alert
                        alert("Please select an option to proceed.");
                    }
                });
            }

            if (isValid && currentStep < steps.length - 1) {
                currentStep++;
                updateWizard();
            }
        });

        prevBtn.addEventListener('click', () => {
            if (currentStep > 0) {
                currentStep--;
                updateWizard();
            }
        });

        // Init
        updateWizard();
    };

    initWizard();

    // --- Loading Animation ---
    const form = document.querySelector('form');
    const submitBtn = document.getElementById('submitBtn');

    if (form) {
        form.addEventListener('submit', (e) => {
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Analyzing...';
            }
        });
    }

});
