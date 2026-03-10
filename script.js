function analyze() {
    const skills = document.getElementById('skills').value;
    const analysis = document.getElementById('analysis');

    if(skills){
        analysis.innerText = `Skills entered: ${skills}`;
    } 
    else {
        analysis.innerText = "Please type your skills or upload your resume.";
    }
}