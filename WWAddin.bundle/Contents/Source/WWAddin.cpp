#define _USE_MATH_DEFINES

#include <Core/CoreAll.h>
#include <Fusion/FusionAll.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <iomanip>
using namespace std;

/*
(C) Copyright 2015 by Autodesk, Inc.

Permission to use, copy, modify, and distribute this software in object code form for any purpose and without fee is hereby granted, 
provided that the above copyright notice appears in all copies and that both that copyright notice and the limited warranty and 
restricted rights notice below appear in all supporting documentation.

AUTODESK PROVIDES THIS PROGRAM "AS IS" AND WITH ALL FAULTS. AUTODESK SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTY OF MERCHANTABILITY OR 
FITNESS FOR A PARTICULAR USE. AUTODESK, INC. DOES NOT WARRANT THAT THE OPERATION OF THE PROGRAM WILL BE UNINTERRUPTED OR ERROR FREE.
*/

/*
License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <atlstr.h>

using namespace adsk::core;
using namespace adsk::fusion;


Ptr<Application> app;
Ptr<UserInterface> ui;
Ptr<Component> newComp;

class EdgeInfo
{
public:
	cv::Point2d _sp;
	cv::Point2d _ep;

	EdgeInfo(double spX, double spY, double epX, double epY)
	{
		_sp.x = spX;
		_sp.y = spY;
		_ep.x = epX;
		_ep.y = epY;
	}
};

#define epsilon 0.00001

class Edges
{
public:
	std::vector<EdgeInfo> mEdges;

	bool areEqual(float a, float b)
	{
		return (fabs(a - b) <= epsilon);
	}

	double minof(double val1, double val2, double val3, double val4)
	{
		return min(min(val1, val2), min(val3, val4));
	}

	double maxof(double val1, double val2, double val3, double val4)
	{
		return max(max(val1, val2), max(val3, val4));
	}

	void AddEdge(EdgeInfo ei)
	{
		double a, b, a1, b1, dp, angle;
		bool check1, check2, check3, check4, check5, check6, check7, check8;

		for (std::vector<EdgeInfo>::iterator it = mEdges.begin(); it != mEdges.end(); ++it)
		{
			EdgeInfo ei1 = *it;

			// Bounding box containment tests for optimization
			if (!(ei._sp.x >= min(ei1._sp.x, ei1._ep.x) && ei._sp.y >= min(ei1._sp.y, ei1._ep.y)) &&
				!(ei._ep.x <= max(ei1._sp.x, ei1._ep.x) && ei._ep.y <= max(ei1._sp.y, ei1._ep.y)) &&
				!(ei1._sp.x >= min(ei._sp.x, ei._ep.x) && ei1._sp.y >= min(ei._sp.y, ei._ep.y)) &&
				!(ei1._ep.x <= max(ei._sp.x, ei._ep.x) && ei1._ep.y <= max(ei._sp.y, ei._ep.y)))
			{
				continue;
			}

			check5 = CheckContainment(ei1, ei._sp);
			check6 = CheckContainment(ei1, ei._ep);
			// Check for overlap
			if (check5 && check6)
			{
				//New Edge fully inside another existing edge. Ignored adding
				return; // New edge fully inside an existing edge, so ignore
			}

			check7 = CheckContainment(ei, ei1._sp);
			check8 = CheckContainment(ei, ei1._ep);
			if (check7 && check8)
			{ // Existing edge full inside the new edge, 
				// so add the new edge and remove the existing one

				mEdges.erase(it);

				AddEdge(ei);

				return;
			}

			if (!check5 && !check6 && !check7 && !check8)
				continue;

			a = (ei._ep.x - ei._sp.x);
			b = (ei._ep.y - ei._sp.y);

			a1 = (ei1._ep.x - ei1._sp.x);
			b1 = (ei1._ep.y - ei1._sp.y);

			dp = (a * a1 + b * b1) / (sqrt(a * a + b * b) * sqrt(a1 * a1 + b1 * b1));
			if (dp > 1)
				dp = 1.0;
			if (dp < -1)
				dp = -1;
			angle = acos(dp);

			if (areEqual(angle, 0.0) || areEqual(angle, M_PI) || areEqual(angle, 2 * M_PI))
			{   // Parallel lines (may or maynot overlap)

				check1 = (areEqual(ei._sp.x, ei1._sp.x) && areEqual(ei._sp.y, ei1._sp.y)); // Start points coincide
				check2 = (areEqual(ei._ep.x, ei1._ep.x) && areEqual(ei._ep.y, ei1._ep.y)); // End points coincide

				if (check1 && check2)
				{
					//New Edge already exists with same SP and EP. Ignored adding 
					return; // Same edge, do not add.
				}

				check3 = (areEqual(ei._sp.x, ei1._ep.x) && areEqual(ei._sp.y, ei1._ep.y)); // Start point and end points coincide
				check4 = (areEqual(ei._ep.x, ei1._sp.x) && areEqual(ei._ep.y, ei1._sp.y));// End point and Start points coincide

				if (check3 && check4)
				{
					//New Edge already exists with SP and EP reversed. Ignored adding 
					return; // Same edge, do not add.
				}

				if (check6 && !check5)
				{// New edge is partially on an existing edge
					// Delete existing edge and add an updated edge
					cv::Point newEP = ei1._ep;
					cv::Point newSP = ei._sp;

					if (check8)
					{
						newEP = ei1._sp;
						newSP = ei._sp;
					}

					mEdges.erase(it);

					AddEdge(EdgeInfo(newSP.x, newSP.y, newEP.x, newEP.y));

					return;
				}

				if (check5 && !check6)
				{// New edge is partially on an existing edge
					// Delete existing edge and add an updated edge
					cv::Point newEP = ei._ep;
					cv::Point newSP = ei1._sp;

					if (check7)
					{
						newSP = ei1._ep;
						newEP = ei._ep;
					}

					mEdges.erase(it);

					AddEdge(EdgeInfo(newSP.x, newSP.y, newEP.x, newEP.y));

					return;
				}
			}
		}

		mEdges.push_back(ei);
	}

	bool CheckContainment(const EdgeInfo &ei, const cv::Point2d &pt)
	{
		if (areEqual(pt.x, ei._sp.x) && areEqual(pt.y, ei._sp.y))
			return true;

		if (areEqual(pt.x, ei._ep.x) && areEqual(pt.y, ei._ep.y))
			return true;

		double px = pt.x;
		double left = min(ei._sp.x, ei._ep.x);
		double right = max(ei._sp.x, ei._ep.x);
		if (px < left || px > right)
			return false;

		double py = pt.y;
		double top = max(ei._sp.y, ei._ep.y);
		double bottom = min(ei._sp.y, ei._ep.y);
		if (py > top || py < bottom)
			return false;

		double a1 = (ei._ep.x - ei._sp.x);
		if (a1 == 0)
			return true;

		double b1 = (ei._ep.y - ei._sp.y);
		double slope = b1 / a1;
		double intercept = ei._sp.y - slope * ei._sp.x;
		if (areEqual(slope * px + intercept, py))
		{
			return true; // point lies on the line
		}
		return false; // outside
	}

	void Clear()
	{
		mEdges.clear();
	}
};

Edges _edges;

double _plateThickness;

const std::string commandId = "ContourGrooveCommandIdCPP";

void createNewComponent()
{
	// Get the active design.
	Ptr<Product> product = app->activeProduct();
	Ptr<Design> design = product;
	if (!design)
		return;
	Ptr<Component> rootComp = design->rootComponent();
	if (!rootComp)
		return;
	Ptr<Occurrences> allOccs = rootComp->occurrences();
	if (!allOccs)
		return;
	Ptr<Occurrence> newOcc = allOccs->addNewComponent(Matrix3D::create());
	if (!newOcc)
		return;
	newComp = newOcc->component();

	Ptr<MaterialLibraries> materialLibraries = app->materialLibraries();

	Ptr<MaterialLibrary> materialLibrary = materialLibraries->itemByName("Fusion 360 Material Library");
	Ptr<Materials> materials = materialLibrary->materials();
	Ptr<Material> material = materials->itemByName("Brass");
	newComp->material(material);
}

Ptr<ExtrudeFeature> createBaseExtrude(Ptr<Profile> prof, double thickness)
{
	if (!newComp)
		return nullptr;
	Ptr<Features> features = newComp->features();
	if (!features)
		return nullptr;
	Ptr<ExtrudeFeatures> extrudes = features->extrudeFeatures();
	if (!extrudes)
		return nullptr;
	Ptr<ExtrudeFeatureInput> extInput = extrudes->createInput(prof, FeatureOperations::JoinFeatureOperation);
	if (!extInput)
		return nullptr;
	Ptr<ValueInput> distance = ValueInput::createByReal(thickness);
	if (!distance)
		return nullptr;
	extInput->setDistanceExtent(false, distance);
	return extrudes->add(extInput);
}

Ptr<ExtrudeFeature> createSlotExtrude(Ptr<ObjectCollection> profiles, double thickness, bool cutType)
{
	if (!newComp)
		return nullptr;
	Ptr<Features> features = newComp->features();
	if (!features)
		return nullptr;
	Ptr<ExtrudeFeatures> extrudes = features->extrudeFeatures();
	if (!extrudes)
		return nullptr;

	Ptr<ExtrudeFeatureInput> extInput;
	Ptr<ValueInput> distance;
	if (cutType)
	{
		extInput = extrudes->createInput(profiles, FeatureOperations::CutFeatureOperation);
		if (!extInput)
			return nullptr;
		distance = ValueInput::createByReal(-1.0 * thickness);
		if (!distance)
			return nullptr;
	}
	else
	{
		extInput = extrudes->createInput(profiles, FeatureOperations::JoinFeatureOperation);
		if (!extInput)
			return nullptr;
		distance = ValueInput::createByReal(1.0 * thickness);
		if (!distance)
			return nullptr;
	}
	extInput->setDistanceExtent(false, distance);
	return extrudes->add(extInput);
}


void createSlotProfile(Ptr<Point3D> sp, Ptr<Point3D> ep, Ptr<SketchLines> &sketchLines, Ptr<SketchArcs> &sketchArcs)
{
	double radius = 0.2;

	double a = ep->x() - sp->x();
	double b = ep->y() - sp->y();
	double modvx = a / sqrt(a * a + b * b);
	double modvy = b / sqrt(a * a + b * b);

	double angle = atan(b / a);
	Ptr<Point3D> pt1 = Point3D::create(sp->x() + radius * sin(angle), sp->y() - radius * cos(angle));
	Ptr<Point3D> pt2 = Point3D::create(ep->x() + radius * sin(angle), ep->y() - radius * cos(angle));
	Ptr<Point3D> pt21 = Point3D::create(ep->x() + radius * modvx, ep->y() + radius * modvy);

	Ptr<Point3D> pt3 = Point3D::create(ep->x() - radius * sin(angle), ep->y() + radius * cos(angle));
	Ptr<Point3D> pt4 = Point3D::create(sp->x() - radius * sin(angle), sp->y() + radius * cos(angle));
	Ptr<Point3D> pt41 = Point3D::create(sp->x() - radius * modvx, sp->y() - radius * modvy);

	sketchLines->addByTwoPoints(pt1, pt2);
	sketchArcs->addByThreePoints(pt2, pt21, pt3);
	sketchLines->addByTwoPoints(pt3, pt4);
	sketchArcs->addByThreePoints(pt4, pt41, pt1);
}

cv::Mat src;
cv::Mat src_gray;
cv::Mat dst;
cv::Mat detected_edges;
int lowThreshold = 50;
int const max_lowThreshold = 100;
int ratio3 = 3;
int kernel_size = 3;
const char* window_name = "Contour groove";

std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;

cv::RNG rng(12345);

static bool showPreview = false;

static void CannyThreshold(int, void*)
{
	try
	{
		if (!detected_edges.empty() && detected_edges.data)
		{
			detected_edges.release();
		}
		detected_edges = cv::Scalar::all(0);
	
	
		blur(src_gray, detected_edges, cv::Size(3, 3));
	
		Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio3, kernel_size);

		findContours(detected_edges, contours, hierarchy, 3/*=RETR_TREE*/, 2/*=CHAIN_APPROX_SIMPLE*/, cv::Point(0, 0));

		dst = cv::Mat::zeros(detected_edges.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
		}

		if (showPreview)
		{
			cv::imshow(window_name, dst);
		}
	}
	catch (exception ex)
	{
		string str = "asd";
	}
}

void cvCleanup()
{
	if (!src.empty() && src.data)
	{
		src.release();
	}

	if (!src_gray.empty() && src_gray.data)
	{
		src_gray.release();
	}

	if (!dst.empty() && dst.data)
	{
		dst.release();
	}

	if (!detected_edges.empty() && detected_edges.data)
	{
		detected_edges.release();
	}
}


void buildContourGroovedPlate(std::string imageFilePath, bool cutType, bool previewRequired)
{
	lowThreshold = 50;
	cvCleanup();

	src = cv::imread(imageFilePath, cv::IMREAD_COLOR); // Read the file
	if (src.empty())
	{
		return;
	}

	if (!src.data) // Check for invalid input
	{
		return;
	}
	
	cv::Size imgSize = src.size();
	int width = imgSize.width;
	int height = imgSize.height;

	if (width * height > 500 * 500)
	{
		ui->messageBox("Please choose a smaller image (width and height < 500 pixels).\nYou may resize the image in an image editor of your choice and then use it with this Addin.", "Information");
		return;
	}

	showPreview = previewRequired;
	if (previewRequired)
	{
		// Create a preview window and associate trackbar with it
		// to control the low Threshold value
		cv::namedWindow(window_name, cv::WINDOW_NORMAL);
		cv::createTrackbar("Threshold", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	}

	// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	// Convert the image to grayscale
	cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

	// Blur the image
	blur(src_gray, detected_edges, cv::Size(3, 3));

	CannyThreshold(0, 0);

	if (previewRequired)
	{
		std::ostringstream stringStream1;
		stringStream1 << "Please adjust the threshold value using the slider in the preview window. After you are convinced with the result, close the preview window to generate the groove.";
		ui->messageBox(stringStream1.str(), "Contour groove", adsk::core::OKButtonType, adsk::core::InformationIconType);

		cv::waitKey(0); // Wait for a keystroke in the window

		std::ostringstream stringStream2;
		if (lowThreshold < 20)
		{
			stringStream2 << "Continue to cut the groove ? Threshold value is set to a low value of " << lowThreshold << ". This can take several minutes to complete.";
		}
		else
		{
			stringStream2 << "Continue to cut the groove ?";
		}
		adsk::core::DialogResults dlgRes = ui->messageBox(stringStream2.str(), "Contour groove", adsk::core::YesNoButtonType, adsk::core::QuestionIconType);
		if (dlgRes == adsk::core::DialogNo)
		{
			return;
		}
	}
	
	// Preview is now ok. So lets get started with the detected edges
	// Check for invalid extracted data
	if (detected_edges.empty()) 
	{
		return;
	}
	if (!detected_edges.data) 
	{
		return;
	}

	_edges.Clear();
	int numContours = contours.size();
	for (int i = 0; i< contours.size(); i++)
	{
		std::vector<cv::Point> contour = contours[i];
		int numPts = contour.size();
		for (int ptIndex = 0; ptIndex < numPts - 1; ptIndex++)
		{
			cv::Point sp = contour[ptIndex];
			cv::Point ep = contour[ptIndex + 1];
			_edges.AddEdge(EdgeInfo(sp.x, height - sp.y, ep.x, height - ep.y));
		}
	}

	int imgWidth = imgSize.width;
	int imgHeight = imgSize.height;

	_plateThickness = imgWidth * 0.05;

	createNewComponent();
	if (!newComp)
	{
		ui->messageBox("New component failed to create", "New Component Failed");
		return;
	}

	Ptr<ConstructionPlane> xzPlane = newComp->xZConstructionPlane();
	if (!xzPlane)
		return;

	Ptr<Sketches> sketches = newComp->sketches();
	if (!sketches)
		return;
	
	Ptr<ExtrudeFeature> extOne = NULL;
	{
		Ptr<Sketch> sketch1 = sketches->add(xzPlane);
		if (!sketch1)
			return;

		sketch1->isComputeDeferred(true);

		Ptr<SketchCurves> curves = sketch1->sketchCurves();
		if (!curves)
			return;
		Ptr<SketchLines> sketchLines = curves->sketchLines();
		if (!sketchLines)
			return;

		sketchLines->addTwoPointRectangle(
			Point3D::create(-2.0 * _plateThickness, -2.0 * _plateThickness, 0.0), 
			Point3D::create(imgWidth + 2.0 * _plateThickness, imgHeight + 2.0 * _plateThickness, 0.0));

		sketch1->isComputeDeferred(false);

		// Create the extrusion.
		Ptr<Profiles> profs = sketch1->profiles();
		if (!profs)
			return;

		Ptr<Profile> prof = profs->item(0);
		if (!prof)
			return;
		extOne = createBaseExtrude(prof, _plateThickness);
		if (!extOne)
			return;
	}
	
	Ptr<ExtrudeFeature> extTwo = NULL;
	{
		// Create a construction Plane
		Ptr<ConstructionPlanes> constructionPlanes = newComp->constructionPlanes();
		Ptr<ConstructionPlaneInput> constructionPlaneInput = constructionPlanes->createInput();
		Ptr<ValueInput> offsetValue = ValueInput::createByReal(_plateThickness);
		if (!offsetValue)
			return;
		constructionPlaneInput->setByOffset(xzPlane, offsetValue);
		Ptr<ConstructionPlane> constructionPlane = constructionPlanes->add(constructionPlaneInput);
		Ptr<Sketch> sketch2 = sketches->add(constructionPlane);
		if (!sketch2)
			return;
		constructionPlaneInput = NULL;

		Ptr<SketchCurves> curves = sketch2->sketchCurves();
		if (!curves)
			return;
		Ptr<SketchLines> sketchLines = curves->sketchLines();
		if (!sketchLines)
			return;
		Ptr<SketchArcs> sketchArcs = curves->sketchArcs();
		if (!sketchArcs)
			return;

		sketch2->isComputeDeferred(true);

		for (std::vector<EdgeInfo>::iterator it = _edges.mEdges.begin(); it != _edges.mEdges.end(); ++it)
		{
			EdgeInfo ei = *it;
			Ptr<Point3D> sp = Point3D::create(ei._sp.x, ei._sp.y);
			Ptr<Point3D> ep = Point3D::create(ei._ep.x, ei._ep.y);
			createSlotProfile(sp, ep, sketchLines, sketchArcs);
		}

		sketch2->isComputeDeferred(false);

		Ptr<Profiles> profs = sketch2->profiles();
		if (!profs)
			return;

		Ptr<ObjectCollection> profiles = ObjectCollection::create();
		if (!profiles)
			return;
		size_t profilesCount = profs->count();

		for (size_t cnt = 0; cnt < profilesCount; cnt++)
		{
			Ptr<Profile> prof = profs->item(cnt);
			if (!prof)
				return;

			Ptr<ProfileLoops> profLoops = prof->profileLoops();
			Ptr<ProfileLoop> profLoop = profLoops->item(0); // Not expecting more than 1 loop for the profiles that we create
			if (!profLoop)
				return;
			Ptr<ProfileCurves> profCurves = profLoop->profileCurves();
			size_t profCurvesCount = profCurves->count();
			if (profCurvesCount > 15)
			{// To avoid inner profiles. If this does not eliminate the profile,
			 // we may need to modify the extrude feature manually after the contour groove is created.
				continue;
			}
			profiles->add(prof);
		}

		if (profilesCount > 0)
		{
			extTwo = createSlotExtrude(profiles, _plateThickness * 0.5, cutType);
			if (!extTwo)
				return;
		}
	}
	
	Ptr<BRepFaces> faces = extOne->faces();
	Ptr<BRepFace> face = faces->item(0);
	Ptr<BRepBody> body = face->body();
	if (!body)
		return;
	std::stringstream ss;
	ss << "Image Groove (" << imageFilePath << " )";
	body->name(ss.str());

	// Zoom to fit view
	Ptr<Viewport> activeVP = app->activeViewport();
	Ptr<Camera> camera = activeVP->camera();
	camera->isFitView(true);
	activeVP->camera(camera);
}

// CommandExecuted event handler.
class OnExecuteEventHander : public adsk::core::CommandEventHandler
{
public:
	void notify(const Ptr<CommandEventArgs>& eventArgs) override
	{
		if (!app)
			return;

		Ptr<Product> product = app->activeProduct();
		if (!product)
			return;
		Ptr<UnitsManager> unitsMgr = product->unitsManager();
		if (!unitsMgr)
			return;

		Ptr<Event> firingEvent = eventArgs->firingEvent();
		if (!firingEvent)
			return;

		Ptr<Command> command = firingEvent->sender();
		if (!command)
			return;

		Ptr<CommandInputs> inputs = command->commandInputs();
		if (!inputs)
			return;

		Ptr<BoolValueCommandInput> previewInput = inputs->itemById("Preview");
		Ptr<DropDownCommandInput> operationInput = inputs->itemById("Operation");
		Ptr<StringValueCommandInput> imageFilePath = inputs->itemById("imageFilePath");
		if (!imageFilePath)
		{
			ui->messageBox("Sorry, Image file path to use was not provided.");
		}
		else
		{
			std::string imageFilePathValue = imageFilePath->value();
			if (!imageFilePathValue.empty())
			{
				Ptr<ListItem> selectedItem = operationInput->selectedItem();
				int index = selectedItem->index();
				bool previewRequired = previewInput->value();
				bool cutType = (index == 0);
				buildContourGroovedPlate(imageFilePathValue, cutType, previewRequired);
			}
		}
	}
};

// CommandCreated event handler.
class CommandCreatedEventHandler : public adsk::core::CommandCreatedEventHandler
{
public:
	void notify(const Ptr<CommandCreatedEventArgs>& eventArgs) override
	{
		if (eventArgs)
		{
			Ptr<Command> cmd = eventArgs->command();
			if (cmd)
			{
				// Connect to the CommandExecuted event.
				Ptr<CommandEvent> onExec = cmd->execute();
				if (!onExec)
					return;
				bool isOk = onExec->add(&onExecuteHander_);
				if (!isOk)
					return;

				// Define the inputs.
				Ptr<CommandInputs> inputs = cmd->commandInputs();
				if (!inputs)
					return;

				HMODULE hm;
				if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPWSTR)&CannyThreshold, &hm))
				{
					WCHAR path[MAX_PATH];
					GetModuleFileNameW(hm, path, sizeof(path));
					PathRemoveFileSpecW(path);
					std::wstring msg2(path);
					std::string msg(msg2.begin(), msg2.end());
					msg += "\\Images\\FusionLogo.jpg";
					inputs->addStringValueInput("imageFilePath", "Image File Path", msg);
				}

				Ptr<DropDownCommandInput> dropDownInputs = inputs->addDropDownCommandInput("Operation", "Operation", adsk::core::DropDownStyles::TextListDropDownStyle);
				dropDownInputs->listItems()->add("Cut", true);
				dropDownInputs->listItems()->add("Join", false);

				inputs->addBoolValueInput("Preview", "Show Preview", true, "", true);
			}
		}
	}
private:
	OnExecuteEventHander onExecuteHander_;
} cmdCreated_;


bool isApplicationStartup(std::string context)
{
	bool res = false;
	std::string::size_type pos = context.find("IsApplicationStartup");
	if (pos != std::string::npos)
	{
		std::string text = context.substr(pos);
		pos = text.find(",");
		if (pos == std::string::npos)
		{
			pos = text.find("}");
		}
		text = text.substr(0, pos);
		res = text.find("true") != std::string::npos;
	}
	return res;
}


extern "C" XI_EXPORT bool run(const char* context)
{
	const std::string commandName = "Contour groove";
	const std::string commandDescription = "Creates a grooved outline from edges identified from an image.";
	const std::string commandResources = "./resources";

	
	app = Application::get();
	if (!app)
		return false;

	ui = app->userInterface();
	if (!ui)
		return false;

	// add a command on create panel in modeling workspace
	Ptr<Workspaces> workspaces = ui->workspaces();
	if (!workspaces)
		return false;
	Ptr<Workspace> modelingWorkspace = workspaces->itemById("FusionSolidEnvironment");
	if (!modelingWorkspace)
		return false;

	Ptr<ToolbarTabs> modelingWorkspaceTabs = modelingWorkspace->toolbarTabs();

	Ptr<ToolbarTab> createTab = modelingWorkspaceTabs->item(0);

	Ptr<ToolbarPanels> toolbarPanels = createTab->toolbarPanels();
	if (!toolbarPanels)
		return false;
	Ptr<ToolbarPanel> toolbarPanel = toolbarPanels->item(0);
	if (!toolbarPanel)
		return false;

	Ptr<ToolbarControls> toolbarControls = toolbarPanel->controls();
	if (!toolbarControls)
		return false;
	Ptr<ToolbarControl> toolbarControl = toolbarControls->itemById(commandId);
	if (toolbarControl)
	{
		ui->messageBox("Contour groove command is already loaded.");
		adsk::terminate();
		return true;
	}
	else
	{
		Ptr<CommandDefinitions> commandDefinitions = ui->commandDefinitions();
		if (!commandDefinitions)
			return false;
		Ptr<CommandDefinition> commandDefinition = commandDefinitions->itemById(commandId);
		if (!commandDefinition)
		{
			commandDefinition = commandDefinitions->addButtonDefinition(commandId, commandName, commandDescription, commandResources);
			if (!commandDefinition)
				return false;
		}

		Ptr<CommandCreatedEvent> commandCreatedEvent = commandDefinition->commandCreated();
		if (!commandCreatedEvent)
			return false;
		commandCreatedEvent->add(&cmdCreated_);
		toolbarControl = toolbarControls->addCommand(commandDefinition);
		if (!toolbarControl)
			return false;
		toolbarControl->isVisible(true);

		if (!isApplicationStartup(context))
			ui->messageBox("Contour Groove is loaded successfully.\r\n\r\nYou can run it from the create panel in modeling workspace.");
	}
	

	return true;
}

extern "C" XI_EXPORT bool stop(const char* context)
{
	if (ui)
	{
		Ptr<CommandDefinitions> commandDefinitions = ui->commandDefinitions();
		Ptr<CommandDefinition> commandDefinition = commandDefinitions->itemById(commandId);
		if (commandDefinition)
		{
			commandDefinition->deleteMe();
		}
		ui = nullptr;
	}
	cvCleanup();
	return true;
}

#ifdef XI_WIN

#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hmodule, DWORD reason, LPVOID reserved)
{
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

#endif // XI_WIN
