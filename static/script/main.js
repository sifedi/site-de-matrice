document.addEventListener('DOMContentLoaded', function () {
    const matrixSizeInput = document.getElementById('matrix-size');
    const addRowButton = document.getElementById('addRowButton');
    const removeRowButton = document.getElementById('removeRowButton');
    const matrixTable = document.getElementById('matrix-table-inner');
    const form = document.getElementById('matrixForm');
    const mainChoiceSelect = document.getElementById('mainChoice');
    const choiceSelect= document.getElementById('method');

    function updateMatrixDimensions() {
        const rowCount = matrixTable.rows.length;
        const colCount = Math.floor(matrixTable.rows[0].cells.length / 2);
        document.getElementById('matrix2-rows').value = rowCount;
        document.getElementById('matrix2-cols').value = colCount;
    }
    
    function updateMatrixSize() {
        const rowCount = matrixTable.rows.length;
        const colCount = matrixTable.rows[0]?.cells.length || 0;
        document.getElementById('matrix-rows').value = rowCount;
        document.getElementById('matrix-cols').value = colCount;
    }


    function setupChoice2Layout() {
        const rows = matrixTable.querySelectorAll('tr');
        rows.forEach((row, rowIndex) => {
            let colIndexA = 0;
            let inputs = row.querySelectorAll('input.matrix-table-input');
            let matrixACount = row.querySelectorAll('td:not(.equal-sign)').length - 1;
    
            inputs.forEach((input, inputIndex) => {
                if (inputIndex < matrixACount) {
                    input.name = `A-${rowIndex}-${colIndexA++}`;
                } else {
                    input.name = `B-${rowIndex}`;
                }
            });
        });
    }
    


    function revertToChoice2Layout() {
        setupChoice2Layout();
    }
   

    function revertToOriginalLayout() {
        
            while (matrixTable.rows.length > 0) {
                matrixTable.deleteRow(0);
            }
    
            for (let i = 0; i < 3; i++) {
                const row = matrixTable.insertRow();
                for (let j = 0; j < 3; j++) {
                    const cell = row.insertCell();
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.className = 'matrix-table-input';
                    cell.appendChild(input);
                }
                const equalsCell = row.insertCell();
                equalsCell.className = 'equal-sign';
                equalsCell.textContent = '=';
    
                const matrixBCell = row.insertCell();
                const input = document.createElement('input');
                input.type = 'text';
                input.className = 'matrix-table-input';
                matrixBCell.appendChild(input);
            }
         
        updateMatrixDimensions();
        //updateMatrixSize(); 
    }
     
    function duplicateMatrixForChoice3() {
        const rows = matrixTable.querySelectorAll('tr');
        const numberOfColumns = rows[0].querySelectorAll('td:not(.equal-sign)').length - 1;
        let matrix = [];
        rows.forEach((row, rowIndex) => {
            let matrixRow = [];
            let cells = row.querySelectorAll('td:not(.equal-sign) .matrix-table-input');
            cells.forEach((input, columnIndex) => {
                if (columnIndex < numberOfColumns) {
                    matrixRow.push(input.value || "0");
                    input.name = `A-${rowIndex}-${columnIndex}`;
                } else {
                    matrixRow.push(input.value || "0");
                    input.name = `B-${rowIndex}-0`; 
                }
            });
    
 
            for (let i = 1; i <= numberOfColumns-1; i++) {
                const newCell = document.createElement('td');
                const input = document.createElement('input');
                input.type = 'text';
                input.className = 'matrix-table-input';
                input.name = `B-${rowIndex}-${i}`;
                newCell.appendChild(input);
                row.appendChild(newCell);
                matrixRow.push("0"); 
            }
    
            matrix.push(matrixRow);
        });
    
        updateMatrixDimensions();
        //updateMatrixSize();
        console.log(matrix);
    }
       
      
    function updateSignAndVisibility() {
        const isChoice2 = mainChoiceSelect.value === 'choice2';
        const isChoice3 = mainChoiceSelect.value === 'choice3';
        const equalSigns = matrixTable.getElementsByClassName('equal-sign');
        for (const equalSign of equalSigns) {
            if (isChoice2 || isChoice3) {
                equalSign.textContent = '*';
            } else {
                equalSign.textContent = '=';
            }
        }
    }
   
    mainChoiceSelect.addEventListener('change', function () {
        revertToOriginalLayout();   
        updateSignAndVisibility();  
        if (mainChoiceSelect.value == 'choice2'){
            revertToChoice2Layout();
        }
        else if (mainChoiceSelect.value == 'choice3') {
            duplicateMatrixForChoice3();           
        } 
        
           
    });

    let matrixAdjustedForSpecialChoices = false;

choiceSelect.addEventListener('change', function () {
    const currentChoice = choiceSelect.value;
    const isSpecialChoice = ['produit_de_matrice_fois_inverse', 'matrix_fois_matrice_transpose'].includes(currentChoice);

    if (isSpecialChoice && !matrixAdjustedForSpecialChoices) {
            matrixAdjustedForSpecialChoices = true;        
            adjustMatrixVisibility();
        
    } else if (!isSpecialChoice && matrixAdjustedForSpecialChoices) 
       {    matrixAdjustedForSpecialChoices = false;
            returnToState();
            
           
        }
    
    updateMatrixDimensions();
    //updateMatrixSize();
});


function updateMatricesForChoice3() {
    let rowCount = matrixTable.rows.length;
    let colCount = (matrixTable.rows[0].cells.length / 2); 

  
for (let i = 0; i < rowCount; i++) {
    let row = matrixTable.rows[i];

    let newCellA = row.insertCell(colCount);
    let newInputA = document.createElement('input');
    newInputA.type = 'text';
    newInputA.className = 'matrix-table-input';
    newInputA.name = `A-${i}-${colCount-0.5}`;
    newCellA.appendChild(newInputA);

    let newCellB = row.insertCell(colCount*2+1); 
    let newInputB = document.createElement('input');
    newInputB.type = 'text';
    newInputB.className = 'matrix-table-input';
    newInputB.name = `B-${i}-${colCount-0.5}`; 
    newCellB.appendChild(newInputB);
}

    const newRow = matrixTable.insertRow(-1);
    for (let j = 0; j < colCount * 2+1; j++) { 
        const newCell = newRow.insertCell(-1);
        const newInput = document.createElement('input');
        newInput.type = 'text';
        newInput.className = 'matrix-table-input';
        newInput.name = (j < colCount) ? `A-${rowCount}-${j}` : `B-${rowCount}-${j-colCount-0.5}`;
        newCell.appendChild(newInput);
    }
    const equalsCell = newRow.insertCell(colCount + 1);
    if (mainChoiceSelect.value == 'choice2') {
        equalsCell.textContent = '*';

    } else if (mainChoiceSelect.value == 'choice3') {
        equalsCell.textContent = '*';
    } else {
        equalsCell.textContent = '=';

    }
    equalsCell.className = 'equal-sign';
    matrixSizeInput.value = parseInt(matrixSizeInput.value) + 1;

    updateMatrixDimensions();
    updateMatrixSize(); 
}





function updateMatricesForChoice3_remove() {
        let rowCount = matrixTable.rows.length;
        let colCount = matrixTable.rows[0].cells.length;
        if ( rowCount > 3 && colCount > 3) {            
            matrixTable.deleteRow(rowCount - 1);               
            for (let i = 0; i < rowCount - 1; i++) { 
                matrixTable.rows[i].deleteCell(colCount - 1);
            }
 
            matrixSizeInput.value = rowCount - 1; 
            matrixTable.classList.remove('new-matrix-added');
        }    
        updateMatrixDimensions();
        updateMatrixSize(); 
    }
    
    function shouldDeleteLastColumn(){
        return choiceSelect.value === 'produit_de_matrice_fois_inverse' || choiceSelect.value === 'matrix_fois_matrice_transpose';
    }
    
    function adjustMatrixVisibility() {
        const shouldBeNbyN = ['produit_de_matrice_fois_inverse', 'matrix_fois_matrice_transpose'].includes(choiceSelect.value);
        const expectedCellsPerRow = shouldBeNbyN ? 3 : getDynamicMatrixSize(); 
    
        if (shouldBeNbyN) {
            while (matrixTable.rows.length > 3) {
                matrixTable.deleteRow(-1);
            }
            while (matrixTable.rows.length < 3) {
                const newRow = matrixTable.insertRow(-1);
                for (let j = 0; j < expectedCellsPerRow; j++) {
                    const newCell = newRow.insertCell(-1);
                    newCell.classList.add('matrix-inputs');
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.className = 'matrix-table-input';
                    input.name = `A-${matrixTable.rows.length - 1}-${j}`;
                    newCell.appendChild(input);
                }
            }
            for (let i = 0; i < matrixTable.rows.length; i++) {
                const row = matrixTable.rows[i];
                while (row.cells.length > expectedCellsPerRow) {
                    row.deleteCell(-1);
                }
                for (let j = row.cells.length; j < matrixTable.rows.length; j++) {
                    const newCell = row.insertCell(-1);
                    newCell.classList.add('matrix-inputs');
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.className = 'matrix-table-input';
                    input.name = `A-${i}-${j}`;
                    newCell.appendChild(input);
                }
            }
        }
        else{
            adjustForDynamicSize(expectedCellsPerRow);
        }
        logMatrixValues();    
        updateMatrixSize();
        updateMatrixDimensions();
    }

    function returnToState() {
        const matrixTable = document.getElementById('matrix-table-inner');
        const existingRowCount = matrixTable.rows.length;
        const existingColCount = matrixTable.rows[0].cells.length;
        if (matrixTable.querySelector('.asterisk-cell')) {
            return;
          }
        for (let i = 0; i < existingRowCount; i++) {
          const asteriskCell = matrixTable.rows[i].insertCell(existingColCount);
          asteriskCell.textContent = '*';
          asteriskCell.classList.add('asterisk-cell');
        }
      
        
        for (let i = 0; i < existingRowCount; i++) {
          for (let j = 0; j < existingRowCount; j++) {
            const newMatrixCell = matrixTable.rows[i].insertCell(existingColCount + j + 1);
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'matrix-table-input';
            newMatrixCell.appendChild(input);
          }
        }
        updateMatrixDimensions();
        updateMatrixSize(); 
      }

    function removeRowAndColumn() {
        let rowCount = matrixTable.rows.length;
        let colCount = matrixTable.rows[0].cells.length;
        let totalAsteriskCells = colCount - (rowCount * 2);
        if (!(rowCount > 3 || totalAsteriskCells > 3)) {
            return;
        }
        if (rowCount > 3) {
            matrixTable.deleteRow(-1);
            rowCount--;
        }
        for (let i = 0; i < rowCount; i++) {
            let row = matrixTable.rows[i];
            row.deleteCell(-1); 
            while(row.cells.length > (rowCount * 2) + 3) {
                row.deleteCell(rowCount * 2);
            }
            row.deleteCell(rowCount); 
        }
    
        updateMatrixDimensions();
        updateMatrixSize(); 
    }
    
    addRowButton.addEventListener('click', function () {
        const currentChoice = choiceSelect.value;
        const isSpecialChoice = ['produit_de_matrice_fois_inverse', 'matrix_fois_matrice_transpose'].includes(currentChoice);
        const issSpecialChoice = ['produit_matrice_demi_bande_inf_largeur_different ', 'multiplication_matrice_demi_bande_inferieur'].includes(currentChoice);
        if (mainChoiceSelect.value !== 'choice3') {
        let rowCount = matrixTable.rows.length;
        let colCount = matrixTable.rows[0].cells.length - 2;

        for (let i = 0; i < rowCount; i++) {
            const cell = matrixTable.rows[i].insertCell(colCount);
            const input = document.createElement('input');
            input.type = 'text';
            input.name = `A-${i}-${colCount}`;
            input.className = 'matrix-table-input';
            cell.appendChild(input);
        }

        const newRow = matrixTable.insertRow(rowCount);
        for (let j = 0; j < colCount + 1; j++) { 
            const newCell = newRow.insertCell(j);
            const newInput = document.createElement('input');
            newInput.type = 'text';
            newInput.name = `A-${rowCount}-${j}`;
            newInput.className = 'matrix-table-input';
            newCell.appendChild(newInput);
        }

        const equalsCell = newRow.insertCell(colCount + 1);
        if (mainChoiceSelect.value == 'choice2') {
            equalsCell.textContent = '*';

        } else if (mainChoiceSelect.value == 'choice3') {
            equalsCell.textContent = '*';
        } else {
            equalsCell.textContent = '=';

        }
        equalsCell.className = 'equal-sign';



        const vectorCell = newRow.insertCell(colCount + 2);
        const vectorInput = document.createElement('input');
        vectorInput.type = 'text';
        vectorInput.name = `B-${rowCount}`;
        vectorInput.className = 'matrix-table-input';
        vectorCell.appendChild(vectorInput);

        matrixSizeInput.value = rowCount + 1;
        updateMatrixDimensions();
      }
        else if (isSpecialChoice){
            let rowCount = matrixTable.rows.length;
            let colCount = matrixTable.rows[0].cells.length;


for (let i = 0; i < rowCount; i++) {
    let row = matrixTable.rows[i];
    let newCellA = row.insertCell(-1);
    let newInputA = document.createElement('input');
    newInputA.type = 'text';
    newInputA.className = 'matrix-table-input';
    newInputA.name = `A-${i}-${colCount}`;
    newCellA.appendChild(newInputA);
}


const newwRow = matrixTable.insertRow(-1);
for (let j = 0; j < colCount + 1; j++) { 
    const newCell = newwRow.insertCell(-1);
    const newInput = document.createElement('input');
    newInput.type = 'text';
    newInput.className = 'matrix-table-input';

    // Determine if the cell is for matrix A or B, assuming the first half of columns belong to A
   // Adjust for new column in A
        newInput.name = `A-${rowCount}-${j}`;
     

    newCell.appendChild(newInput);
}
updateMatrixSize();
updateMatrixDimensions();                 
            }
            else if (issSpecialChoice){
                updateMatricesForChoice3();

            }
            
        updateMatrixDimensions();
    });


    removeRowButton.addEventListener('click', function () {
        const currentChoice = choiceSelect.value;
        const isSpecialChoice = ['produit_de_matrice_fois_inverse', 'matrix_fois_matrice_transpose'].includes(currentChoice);
        const issSpecialChoice = ['produit_matrice_demi_bande_inf_largeur_different ', 'multiplication_matrice_demi_bande_inferieur'].includes(currentChoice);
            if (mainChoiceSelect.value !== 'choice3'){
            let rowCount = matrixTable.rows.length;
            if (rowCount > 3) {
                for (let i = 0; i < rowCount - 1; i++) {
                    let cellsInRow = matrixTable.rows[i].cells.length;
                    matrixTable.rows[i].deleteCell(cellsInRow - 3);
                }

                matrixTable.deleteRow(-1);

                matrixSizeInput.value = rowCount - 1; 
            }


        }
        else if(isSpecialChoice){
            updateMatricesForChoice3_remove()
        }
        else if(issSpecialChoice){
            removeRowAndColumn();
        }

});
    
form.addEventListener('submit', function (event) {
    event.preventDefault(); 
    if (mainChoiceSelect.value === 'choice2') {
        setupChoice2Layout();
    }
    const formData = new FormData(this);
    console.log(Array.from(formData.entries()));
    fetch('/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json())

        .then(data => {
            const resultContainer = document.getElementById('result');
            resultContainer.innerHTML = '';

            if (data.error_message) {
                alert("⚠ Erreur : " + data.error_message);

            } else {
                if (choiceSelect.value === 'LU_dense' || choiceSelect.value==='LU_bande'){
                    const lMatrixTable = document.createElement('table');
            lMatrixTable.className = 'result-matrix-table';
            data.L.forEach(row => {
                const tr = document.createElement('tr');
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = Number(cell).toFixed(2);
                    tr.appendChild(td);
                });
                lMatrixTable.appendChild(tr);
            });
            resultContainer.appendChild(document.createTextNode('Matrice L ='));
            resultContainer.appendChild(lMatrixTable);

            // Display U matrix
            const uMatrixTable = document.createElement('table');
            uMatrixTable.className = 'result-matrix-table';
            data.U.forEach(row => {
                const tr = document.createElement('tr');
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = Number(cell).toFixed(2);
                    tr.appendChild(td);
                });
                uMatrixTable.appendChild(tr);
            });
            resultContainer.appendChild(document.createTextNode('Matrice U ='));
            resultContainer.appendChild(uMatrixTable);
            if (data.solution) {
                const solutionTable = document.createElement('table');
                solutionTable.className = 'result-matrix-table';
                data.solution.forEach(cell => {
                    const tr = document.createElement('tr');
                    const td = document.createElement('td');
                    td.textContent = Number(cell).toFixed(2);
                    tr.appendChild(td);
                    solutionTable.appendChild(tr);
                });
            
                resultContainer.appendChild(document.createTextNode('La Solution X ='));
                resultContainer.appendChild(solutionTable);
            }
            

                }
                if (mainChoiceSelect.value === 'choice3') {
                    if (choiceSelect.value==='matrix_fois_matrice_transpose' || choiceSelect.value==='produit_de_matrice_fois_inverse'){
                        const mMatrixTable = document.createElement('table');
            mMatrixTable.className = 'result-matrix-table';
            data.m1.forEach(row => {
                const tr = document.createElement('tr');
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = Number(cell).toFixed(2);
                    tr.appendChild(td);
                });
                mMatrixTable.appendChild(tr);
            });
            if(choiceSelect.value==='matrix_fois_matrice_transpose'){
                resultContainer.appendChild(document.createTextNode('La Transpsoée de la Matrice ='));
            }
            if(choiceSelect.value==='produit_de_matrice_fois_inverse'){
                resultContainer.appendChild(document.createTextNode('L\' inverse de la Matrice ='));
            }
            resultContainer.appendChild(mMatrixTable);
            
            if (data.solution) {
                const solutionTable = document.createElement('table');
                solutionTable.className = 'result-matrix-table';
                data.solution.forEach(row => {
                    const tr = document.createElement('tr');
                    row.forEach(cell => {
                        const td = document.createElement('td');
                        td.textContent = Number(cell).toFixed(2);
                        tr.appendChild(td);
                    });
                    solutionTable.appendChild(tr);
                });
            
                resultContainer.appendChild(document.createTextNode('La Solution X ='));
                resultContainer.appendChild(solutionTable);
            }

                    }
                    const matrixTable = document.createElement('table');
                    matrixTable.className = 'result-matrix-table';
    
                    data.result.forEach(row => {
                        const tr = document.createElement('tr');
                        row.forEach(cell => {
                            const td = document.createElement('td');
                            td.textContent = Number(cell).toFixed(2);
                            tr.appendChild(td);
                        });
                        matrixTable.appendChild(tr);
                    });
    
                    resultContainer.appendChild(matrixTable);
                }
                 else {
                    if (data.result && !data.L && !data.U && !data.m1){
                const textElement = document.createElement('div');
                textElement.textContent = 'La solution X = ';
                resultContainer.appendChild(textElement);

                const matrixResult = document.createElement('div');
                matrixResult.className = 'matrix-result';

                data.result.forEach(value => {
                    const numberElement = document.createElement('div');
                    numberElement.className = 'matrix-number';
                    numberElement.textContent = Number(value).toFixed(2);
                    matrixResult.appendChild(numberElement);
                });

                resultContainer.appendChild(matrixResult);
            }
        }
        }
        })
        .catch(error => {
            console.error('Error:', error);
        });

});

    
});
document.addEventListener('DOMContentLoaded', function () {
    const mainChoiceSelect = document.getElementById('mainChoice');
    const methodSelect = document.getElementById('method');
    const bandwidthInputDiv = document.getElementById('bandwidth-input');

    const methodsForChoice = {
        'choice1': [{
                text: 'Triangulaire avec une matrice supérieure',
                value: 'res_sup_dense',
                description: 'Votre matrice doit être triangulaire supérieure .'
            },
            {
                text: 'Triangulaire avec une matrice inférieure',
                value: 'res_inf_dense',
                description: 'Votre matrice doit être triangulaire inférieure .'
            },
            {
                text: 'Triangulaire avec une matrice demi-bande supérieure',
                value: 'res_demi_bande_sup',
                description: 'Votre matrice doit être triangulaire demi-bande supérieure .'

            },
            {
                text: 'Triangulaire avec une matrice demi-bande inférieure',
                value: 'res_demi_bande_inf',
                description: 'Votre matrice doit être triangulaire demi-bande inférieure .'
            },
            {
                text: 'Avec une matrice dense avec la méthode de la décomposition LU',
                value: 'LU_dense',
                description: 'Votre matrice doit être dense et symétrique définie positive .'
            },
            {
                text: 'Avec une matrice bande avec la méthode de la décomposition LU',
                value: 'LU_bande',
                description: 'Votre matrice doit être bande et symétrique définie positive .'
            },
            {
                text: 'Avec une matrice dense avec la méthode de Gauss avec pivotage',
                value: 'Gauss_pivotage_dense',
                description: 'Votre matrice doit être dense et NON symétrique .'

            },
            {
                text: 'Avec une matrice bande avec la méthode de Gauss avec pivotage',
                value: 'Gauss_pivotage_bande',
                description: 'Votre matrice doit être bande et NON symétrique .'
            },
            {
                text: 'Avec une matrice dense avec la méthode de Gauss',
                value: 'Gauss_dense',
                description: 'Votre matrice doit être dense et symétrique définie positive .'
            },
            {
                text: 'Avec une matrice bande avec la méthode de Gauss',
                value: 'Gauss_bande',
                description: 'Votre matrice doit être bande et symétrique définie positive .'

            },
            {
                text: 'Avec une matrice dense avec la méthode de Gauss-Jordan',
                value: 'gauss_jordan',
                description: 'Votre matrice doit être dense et symétrique définie positive .'

            },
            {
                text: 'Avec une matrice dense avec la méthode de Cholesky',
                value: 'Cholesky_dense',
                description: 'Votre matrice doit être dense .'

            },
            {
                text: 'Avec une matrice bande avec la méthode de Cholesky',
                value: 'Cholesky_bande',
                description: 'Votre matrice doit être bande .'

            },

            {
                text: 'Avec une matrice dense avec la méthode de Jacobi avec ε',
                value: 'jacobi_with_epsilon',
                description: 'Votre matrice doit être dense .'

            },
            {
                text: 'Avec une matrice dense avec la méthode de Jacobi avec le maximum d\'itérations',
                value: 'jacobi_with_max_iteration',
                description: 'Votre matrice doit être dense .'

            },
            {
                text: 'Avec une matrice dense avec la méthode de Gauss-Seidel avec ε',
                value: 'gauss_seidel_with_epsilon',
                description: 'Votre matrice doit être dense .'
            },
            {
                text: 'Avec une matrice dense avec la méthode de Gauss-Seidel avec le maximum d\'itérations',
                value: 'gauss_seidal_with_max_iteration',
                description: 'Votre matrice doit être dense .'
            }
            

        ],

        'choice2': [{
                text: 'Matrice dense',
                value: 'produit_matrice_vecteur',
                description: 'Votre matrice doit être dense .'
            },
            {
                text: 'Matrice triangulaire inférieure dense',
                value: 'produit_matrice_triangulaire_inferieure_vecteur',
                description: 'Votre matrice doit être dense et triangulaire inférieure .'

            },
            {
                text: 'Matrice triangulaire supérieure dense',
                value: 'produit_matrice_triangulaire_superieure_vecteur',
                description: 'Votre matrice doit être dense et triangulaire supérieure .'

            },
            {
                text: 'Matrice triangulaire inférieure demi-bande',
                value: 'produit_matrice_triangulaire_inferieure_demi_bande_vecteur',
                description: 'Votre matrice doit être triangulaire demi-bande inférieure .'

            },
            {
                text: 'Matrice triangulaire supérieure demi-bande',
                value: 'produit_matrice_demi_bande_superieure_vecteur',
                description: 'Votre matrice doit être triangulaire demi-bande supérieure .'
            }
        ], 
        'choice3': [{
                text: 'Matrice bande de largeur totale 2m + 1 × Matrice demi-bande inférieure de largeur m + 1',
                value: 'multiplication_matrice_demi_bande_inferieur',
                description: 'Votre première matrice doit être bande de largeur totale 2m + 1 ET votre deuxième matrice doit être demi-bande inférieure de largeur m + 1 .'
            },
            {
                text: 'Matrice demi-bande inférieure de largeur s + 1 × Matrice demi-bande supérieure de largeur r + 1',
                value: 'produit_matrice_demi_bande_inf_largeur_different ',
                description: 'Votre première matrice doit être demi-bande inférieure de largeur S + 1 ET votre deuxième matrice doit être demi-bande supérieure de largeur R + 1  ( NB: R et S doivent être différents ) .'

            },
            {
                text: 'Matrice bande de largeur totale 2m + 1 × Matrice inverse',
                value: 'produit_de_matrice_fois_inverse',
                description: 'Votre matrice doit être bande  ( NB: l\'inverse de votre matrice sera calculé automatiquement ) .'
                
            },
            {
                text: 'Matrice bande de largeur totale 2m + 1 × Matrice transposée',
                value: 'matrix_fois_matrice_transpose',
                description: 'Votre matrice doit être bande  ( NB: la transposée de votre matrice sera calculée automatiquement ) .'

            }
        ] 
    };

    function showOrHideBandwidthInput() {
        const selectedMethodOption = methodSelect.options[methodSelect.selectedIndex];
        const selectedMethodValue = selectedMethodOption.value;
    
        const bandwidthInputDiv = document.getElementById('bandwidth-input');
        const bandwidthInputDivS = document.getElementById('bandwidth-input-s');
        const bandwidthInputDivR = document.getElementById('bandwidth-input-r');
        const bandwidthInputDivM = document.getElementById('bandwidth-input-m');
        const bandwidthInputDivE = document.getElementById('bandwidth-input-e');
    
        if (selectedMethodValue.toLowerCase().includes( "different")) {
            bandwidthInputDiv.style.display = 'none'; 
            bandwidthInputDivS.style.display = ''; 
            bandwidthInputDivR.style.display = '';
            bandwidthInputDivM.style.display = 'none'; 
            bandwidthInputDivE.style.display = 'none';
        } else if (selectedMethodOption.text.toLowerCase().includes("bande") || selectedMethodOption.text.toLowerCase().includes("demi-bande")) {
            bandwidthInputDiv.style.display = ''; 
            bandwidthInputDivS.style.display = 'none'; 
            bandwidthInputDivR.style.display = 'none';
            bandwidthInputDivM.style.display = 'none'; 
            bandwidthInputDivE.style.display = 'none'; 
        } else if (selectedMethodOption.text.includes("ε") ){ 
            bandwidthInputDiv.style.display = 'none'; 
            bandwidthInputDivS.style.display = 'none';
            bandwidthInputDivR.style.display = 'none';
            bandwidthInputDivM.style.display = 'none'; 
            bandwidthInputDivE.style.display = ''; 
        } else if (selectedMethodOption.text.toLowerCase().includes("itérations")){
            bandwidthInputDiv.style.display = 'none';
            bandwidthInputDivS.style.display = 'none';
            bandwidthInputDivR.style.display = 'none';
            bandwidthInputDivM.style.display = ''; 
            bandwidthInputDivE.style.display = 'none'; 

        }
         else {
            bandwidthInputDiv.style.display = 'none';
            bandwidthInputDivS.style.display = 'none';
            bandwidthInputDivR.style.display = 'none';
            bandwidthInputDivM.style.display = 'none'; 
            bandwidthInputDivE.style.display = 'none';
        }
    }

    function updateDescription() {
        const selectedMethodOption = methodSelect.options[methodSelect.selectedIndex];
        if (selectedMethodOption) {
            // Find the method in the methodsForChoice object to get its description
            const selectedChoice = mainChoiceSelect.value;
            const methodDetails = methodsForChoice[selectedChoice].find(method => method.value === selectedMethodOption.value);
            if (methodDetails && methodDetails.description) {
                description.textContent = methodDetails.description; // Use the description from the method object
            } else {
                description.textContent = ''; // Clear the description if not found
            }
        } else {
            description.textContent = ''; // Clear the description if no option is selected
        }
    }
    
    


    function initializeMethodOptions() {
        const selectedChoice = mainChoiceSelect.value;

        while (methodSelect.firstChild) {
            methodSelect.removeChild(methodSelect.firstChild);
        }

        methodsForChoice[selectedChoice].forEach(function (method) {
            const option = document.createElement('option');
            option.value = method.value;
            option.textContent = method.text;
            methodSelect.appendChild(option);
        });
        showOrHideBandwidthInput();
        updateDescription(); 
    }


    mainChoiceSelect.addEventListener('change', initializeMethodOptions);
    //methodSelect.addEventListener('change', showOrHideBandwidthInput)
    methodSelect.addEventListener('change', function() {
        showOrHideBandwidthInput();
        updateDescription(); // Update description when a new method is selected
    });
    initializeMethodOptions();

});