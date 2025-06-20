document.addEventListener('DOMContentLoaded', () => {
    const stockStatesContainer = document.getElementById('stock-states-container');
    const tradeLogTableBody = document.getElementById('trade-log-table-body');
    const lastUpdatedSpan = document.getElementById('last-updated');
    const loadingMessage = document.getElementById('loading-message');
    const errorMessage = document.getElementById('error-message');
   
    const monitorBotBtn = document.getElementById('monitorBotBtn');
    const stopMonitorBtn = document.getElementById('stopMonitorBtn');
    const statusModal = document.getElementById('statusModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalMessage = document.getElementById('modalMessage');
    const closeModalBtn = document.getElementById('closeModalBtn');

    const API_BASE_URL = '';

    function showModal(title, message, isError = false) {
        modalTitle.textContent = title;
        modalMessage.innerHTML = message;
        if (isError) {
            modalMessage.classList.add('text-red-600');
        } else {
            modalMessage.classList.remove('text-red-600');
        }
        statusModal.classList.remove('hidden');
    }

    function hideModal() {
        statusModal.classList.add('hidden');
    }

    async function callBackend(endpoint, method = 'POST') {
        try {
            const response = await fetch(`${API_BASE_URL}/api/${endpoint}`, { method: method });
            const data = await response.json();

            if (response.ok) {
                showModal('Success', `<span class="text-green-600 font-semibold">${data.message}</span>`);
            } else {
                showModal('Error', `<span class="text-red-600 font-semibold">${data.message}</span>`, true);
            }
        } catch (error) {
            console.error(`Error calling backend endpoint ${endpoint}:`, error);
            showModal('Network Error', `Could not connect to the backend server. Is the Flask server running? Error: ${error.message}`, true);
        }
    }

    monitorBotBtn.addEventListener('click', () => {
        showModal('Starting Monitor...', 'Sending request to the backend. The dashboard will start updating shortly.');
        callBackend('monitor');
    });

    stopMonitorBtn.addEventListener('click', () => {
        showModal('Stopping Monitor...', 'Sending request to the backend.');
        callBackend('stop_monitor');
    });

    closeModalBtn.addEventListener('click', hideModal);
    statusModal.addEventListener('click', (event) => {
        if (event.target === statusModal) {
            hideModal();
        }
    });


    async function fetchData() {
        if (statusModal.classList.contains('hidden')) {
            loadingMessage.classList.remove('hidden');
            errorMessage.classList.add('hidden');
            stockStatesContainer.innerHTML = '';
            tradeLogTableBody.innerHTML = '';
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/output.json`);
            
            if (!response.ok) {
                if (response.status === 404) {
                    errorMessage.classList.remove('hidden');
                    errorMessage.querySelector('p:first-child').textContent = 'output.json not found.';
                    errorMessage.querySelector('p:last-child').innerHTML = 'Please ensure the bot is running and generating the file.';
                    lastUpdatedSpan.textContent = 'N/A';
                    stockStatesContainer.innerHTML = '<p class="text-center text-gray-500 col-span-full">Waiting for bot data...</p>';
                    tradeLogTableBody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-gray-500">Waiting for bot data...</td></tr>';
                    return;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            lastUpdatedSpan.textContent = data.timestamp || 'N/A';

            if (data.stock_states && Object.keys(data.stock_states).length > 0) {
                let stockStatesHtml = '';
                for (const ticker in data.stock_states) {
                    const state = data.stock_states[ticker];
                    const inPosition = state.in_buy_position ? 'Yes' : 'No';
                    const signalColor = state.last_ai_signal === 'BUY (AI)' ? 'text-green-600' :
                                        state.last_ai_signal === 'SELL (AI)' ? 'text-red-600' : 'text-yellow-600';
                    const positionBg = state.in_buy_position ? 'bg-green-50' : 'bg-blue-50';

                    stockStatesHtml += `
                        <div class="bg-white ${positionBg} p-4 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow">
                            <h3 class="text-xl font-bold text-gray-700 mb-2">${ticker}</h3>
                            <p class="text-sm text-gray-600">Current Price: <span class="font-medium text-gray-800">${state.current_price !== null ? state.current_price.toFixed(2) : 'N/A'}</span></p>
                            <p class="text-sm text-gray-600">AI Signal: <span class="font-medium ${signalColor}">${state.last_ai_signal}</span></p>
                            <p class="text-sm text-gray-600">In Position: <span class="font-medium">${inPosition}</span></p>
                            ${state.in_buy_position && state.entry_price !== null ? `<p class="text-sm text-gray-600">Entry Price: <span class="font-medium">${state.entry_price.toFixed(2)}</span></p>` : ''}
                            ${state.in_buy_position && state.stop_loss_price !== null ? `<p class="text-sm text-gray-600">Stop Loss: <span class="font-medium">${state.stop_loss_price.toFixed(2)}</span></p>` : ''}
                            ${state.in_buy_position && state.high_since_entry !== null ? `<p class="text-sm text-gray-600">High Since Entry: <span class="font-medium">${state.high_since_entry.toFixed(2)}</span></p>` : ''}
                        </div>
                    `;
                }
                stockStatesContainer.innerHTML = stockStatesHtml;
            } else {
                stockStatesContainer.innerHTML = '<p class="text-center text-gray-500 col-span-full">No stock states available yet.</p>';
            }

            if (data.trade_log && data.trade_log.length > 0) {
                let tradeLogHtml = '';
                const sortedTradeLog = data.trade_log.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

                sortedTradeLog.forEach(trade => {
                    const plClass = trade.p_l_percent !== null ? (trade.p_l_percent >= 0 ? 'positive-pl' : 'negative-pl') : '';
                    tradeLogHtml += `
                        <tr class="hover:bg-gray-50">
                            <td class="px-4 py-2 text-sm text-gray-700">${trade.timestamp}</td>
                            <td class="px-4 py-2 text-sm text-gray-700 font-semibold">${trade.ticker}</td>
                            <td class="px-4 py-2 text-sm text-gray-700">${trade.type}</td>
                            <td class="px-4 py-2 text-sm text-gray-700">${trade.entry_price !== null ? trade.entry_price.toFixed(2) : 'N/A'}</td>
                            <td class="px-4 py-2 text-sm text-gray-700">${trade.exit_price !== null ? trade.exit_price.toFixed(2) : 'N/A'}</td>
                            <td class="px-4 py-2 text-sm ${plClass}">${trade.p_l_percent !== null ? trade.p_l_percent.toFixed(2) + '%' : 'N/A'}</td>
                        </tr>
                    `;
                });
                tradeLogTableBody.innerHTML = tradeLogHtml;
            } else {
                tradeLogTableBody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-gray-500">No trades yet...</td></tr>';
            }

        } catch (error) {
            console.error('Error fetching data:', error);
            errorMessage.classList.remove('hidden');
            errorMessage.querySelector('p:first-child').textContent = 'Error loading data.';
            errorMessage.querySelector('p:last-child').innerHTML = `A network or server error occurred. Ensure the Flask backend is running.`;
            stockStatesContainer.innerHTML = '';
            tradeLogTableBody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-red-500">Failed to load trade log due to backend error.</td></tr>';
        } finally {
            loadingMessage.classList.add('hidden');
        }
    }


    fetchData();

    // every 10 seconds refresh
    setInterval(fetchData, 10000); 
});
