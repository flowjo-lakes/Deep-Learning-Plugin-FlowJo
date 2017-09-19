import java.awt.Dimension;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import javax.swing.*;

import com.treestar.flowjo.engine.utility.ClusterPrompter;
import com.treestar.flowjo.engine.utility.ParameterOptionHolder;
import com.treestar.lib.FJPluginHelper;
import com.treestar.lib.core.ExportFileTypes;
import com.treestar.lib.core.ExternalAlgorithmResults;
import com.treestar.lib.core.PopulationPluginInterface;
import com.treestar.lib.file.FileUtil;
import com.treestar.lib.gui.GuiFactory;
import com.treestar.lib.gui.numberfields.RangedIntegerTextField;
import com.treestar.lib.gui.panels.FJLabel;
import com.treestar.lib.gui.swing.SwingUtil;
import com.treestar.lib.xml.SElement;

import com.treestar.lib.FJPluginHelper;



public class DeepLearningFlowJo implements PopulationPluginInterface
{
	private List<String> fParameters = new ArrayList<String>();
	
	//arguments to export to python script
	private String targetPath;
	private String sourcePath;
	private String outputPath;
	private String resultName;
	
	//ImportCSV argument
	private SElement queryElement;
	
	//python script place holder
	private static File gScriptFile = null;
	
	//default min epochs
	private int numEpochs = 1;
	
	//start off with an empty state
	private pluginState state = pluginState.empty; 
	
	// This enum defines the possible states of the plugin node
	public enum pluginState {
		empty, learned, ready
	}
	
	@Override
	public boolean promptForOptions(SElement fcmlQueryElement, List<String> parameterNames) {
		//only run this method when the plugin is initialized 
		if (state != pluginState.empty)
			return true;	
		//obtain the fcmlQueryElement for the ImportCSV method
		queryElement = fcmlQueryElement;
        
        List<Object> guiObjects = new ArrayList<Object>();
		FJLabel explainText = new FJLabel();
		guiObjects.add(explainText);

		explainText = new FJLabel();
		guiObjects.add(explainText);
		String text = "<html><body>";
		text += "Enter the number of Epochs";
		text += "</body></html>";
		explainText.setText(text);
		// Epoch entry
		FJLabel label = new FJLabel("Number of Epochs (1 - 600) ");
		String tip = "A higher number of Epochs will result in more accurately trained data but takes longer.";
		label.setToolTipText(tip);
		RangedIntegerTextField epochInputField = new RangedIntegerTextField(1, 600);
		epochInputField.setInt(numEpochs);
		epochInputField.setToolTipText(tip);
		GuiFactory.setSizes(epochInputField, new Dimension(50, 25));
		Box box = SwingUtil.hbox(Box.createHorizontalGlue(), label, epochInputField, Box.createHorizontalGlue());
		guiObjects.add(box);

		SElement algorithmElement = getElement();
        // pass the XML element to the cluster prompter
		ParameterOptionHolder prompter = new ParameterOptionHolder(algorithmElement);
        if (!prompter.promptForOptions(algorithmElement, parameterNames))
            return false;
        algorithmElement = prompter.getElement();
        setElement(algorithmElement);

		int option = JOptionPane.showConfirmDialog(null, guiObjects.toArray(), "Deep_Learning_Plugin",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE, null);
		
		if (option == JOptionPane.OK_OPTION) 
		{
			
			// user clicked ok, get all selected parameters
			fParameters.clear();
			SElement prEl = prompter.getElement();
			for (SElement child : prEl.getChildren()) {
				if (child.getName().compareTo("Parameter") == 0) {
					String name = child.getString("name");
					int idx = name.indexOf(" ::");
					if (idx >= 0)
						name = name.substring(0, idx);
					fParameters.add(child.getString("name"));
				}
			}
			
			// get other GUI inputs
			numEpochs = epochInputField.getInt();
			return true;
		} 
		else
			return false;
	}
	
	@Override
	public SElement getElement() {
		SElement result = new SElement(getName());
		// store the parameters the user selected
		if (!fParameters.isEmpty()) {
			SElement elem = new SElement("Parameters");
			result.addContent(elem);
			for (String pName : fParameters) {
				SElement e = new SElement("P");
				e.setString("name", pName);
				elem.addContent(e);
			}
		}
		result.setInt("numEpochs", numEpochs);
		result.setString("state", state.toString());
		if (state == pluginState.learned)
		{
			result.setString("resultName", resultName.toString());
			result.setString("sourcePath", sourcePath.toString());
		}
		return result;	
	}

	@Override
	public Icon getIcon() { return null; }

	@Override
	public String getName() { return "Deep_Learning_Plugin"; }

	@Override
	public List<String> getParameters() {
		return fParameters;
	}
	@Override
	public String getVersion()
	{
		return "1.0";
	}
	
	/****************************************************************
	* Function: getScriptFile
	* Purpose: Unpacks the python script from the jar file saves
	* 	it to the local files system in the outputFolder
	* Arguments: Path to the output folder defined by FlowJo
	* Result: Returns the file object of the python script
	****************************************************************/
	private File getScriptFile(File absolutePath) {
		if(gScriptFile == null) 
		{
		    InputStream findScriptPath = this.getClass().getClassLoader().getResourceAsStream("python/train_MMD_ResNet.py");
		    if(findScriptPath != null) 
		    {
		    	try 
		    	{
	            File scriptFile = new File(absolutePath, "train_MMD_ResNet.py");
	            FileUtil.copyStreamToFile(findScriptPath, scriptFile);
	            gScriptFile = scriptFile;
		    	} 
		    	catch (Exception exception) 
		    	{
		    		System.out.println("Script not found");
		    	}
	        System.out.println("Script found");
	    	}
		}	
		return gScriptFile;
	}
	
	/****************************************************************
	* Function: executePython
	* Purpose: Executes the Deep Learning python script and prints
	* the results
	* Arguments: None
	* Result: Returns true if everything went as expected, false if
	* any exceptions occur
	****************************************************************/
	public boolean executePython(File outputFolder)
	{
		System.out.println(outputFolder);
		try 
		{	
			File myPythonFile = getScriptFile(outputFolder);
			System.out.println("Trying to execute python script....\n");			
			
			if (myPythonFile != null)
			{
				String execLine =   
					"python" + " \""
					+ myPythonFile.getAbsolutePath() + "\" "
					+ numEpochs + " \""
					+ sourcePath + "\" \""
					+ targetPath + "\" \""
					+ outputFolder + "\" "
					+ resultName + " ";			
				Process proc = Runtime.getRuntime().exec(execLine);
				System.out.println("Working.....");
				
				//prepare to deliver the output from the python file
				OutputStream stdout = proc.getOutputStream();
				InputStream stdin = proc.getInputStream();
				InputStream stderr = proc.getErrorStream();
				InputStreamReader isrIn = new InputStreamReader(stdin);
				InputStreamReader isr = new InputStreamReader(stderr);
				BufferedReader br = new BufferedReader(isrIn);
				
	            Thread.sleep(1000);

	            //deliver the output from the python file
	            String line = null;
	            while ((line = br.readLine()) != null) {
	                System.out.println(line);
	            }
	            //wait for the process to finish up
				proc.waitFor();
				
				System.out.println("Execution successful!\n");
				
				return true;
			}
		}
		catch (InterruptedException e) 
		{
			e.printStackTrace();
			return false;
		}
		catch (IOException e)
		{
			e.printStackTrace();
			return false;
		} 
		return false;
	}

	/**
	 * Invokes the algorithm and returns the results.
	 */
	@Override
	public ExternalAlgorithmResults invokeAlgorithm(SElement fcmlElem, File sampleFile, File outputFolder) {
		ExternalAlgorithmResults results = new ExternalAlgorithmResults();
		if (state == pluginState.empty)
		{
			sourcePath = sampleFile.getAbsolutePath();
			String fileName = sampleFile.getName().replace(' ', '_');
			resultName = fileName.replace("..ExtNode.csv", "");
			state = pluginState.learned;
		}
		//second call
		else if (state == pluginState.learned)
		{
			targetPath = sampleFile.getAbsolutePath();
			//the ready state prevents users from continuing to use the plugin outside it's intended use
			state = pluginState.ready;
			outputPath = outputFolder.getAbsolutePath();
			
			if (executePython(outputFolder)) {		
				//Import the resultant CSV file into the current workspace
				FJPluginHelper.loadSamplesIntoWorkspace(fcmlElem, new String[]{
						outputFolder + "/" + resultName + "_" + numEpochs + "E_" + "DL.csv",});
			}
		}
		return results;
	}

	@Override
	public void setElement(SElement element) { 
		SElement params = element.getChild("Parameters");
		if (params == null)
			return;
		fParameters.clear();
		for (SElement elem : params.getChildren()) {
			fParameters.add(elem.getString("name"));
		}
 
		numEpochs = element.getInt("numEpochs");
		state = pluginState.valueOf(element.getString("state"));
		if (state == pluginState.learned)
		{
			sourcePath = element.getString("sourcePath");
			resultName = element.getString("resultName");
		}
	}

	@Override
	public ExportFileTypes useExportFileType()
	{
		return ExportFileTypes.CSV_SCALE;
	}
}
