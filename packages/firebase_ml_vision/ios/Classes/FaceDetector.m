// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#import "FLTFirebaseMlVisionPlugin.h"

@interface FaceDetector ()
@property MLKFaceDetector *detector;
@end

@implementation FaceDetector
- (instancetype)initWithVision:(FIRVision *)vision options:(NSDictionary *)options {
  self = [super init];
  if (self) {
    _detector = [MLKFaceDetector faceDetectorWithOptions:options]:[FaceDetector parseOptions:options]];
  }
  return self;
}

- (void)handleDetection:(MLKVisionImage *)image result:(FlutterResult)result {
  [_detector
      processImage:image
        completion:^(NSArray<MLKFace *> *_Nullable faces, NSError *_Nullable error) {
          if (error) {
            [FLTFirebaseMlVisionPlugin handleError:error result:result];
            return;
          } else if (!faces) {
            result(@[]);
            return;
          }

          NSMutableArray *faceData = [NSMutableArray array];
          for (MLKFace *face in faces) {
            id smileProb = face.hasSmilingProbability ? @(face.smilingProbability) : [NSNull null];
            id leftProb = face.hasLeftEyeOpenProbability ? @(face.leftEyeOpenProbability) : [NSNull null];
            id rightProb = face.hasRightEyeOpenProbability ? @(face.rightEyeOpenProbability) : [NSNull null];

            NSDictionary *data = @{
              @"left" : @(face.frame.origin.x),
              @"top" : @(face.frame.origin.y),
              @"width" : @(face.frame.size.width),
              @"height" : @(face.frame.size.height),
              @"headEulerAngleY" : face.hasHeadEulerAngleY ? @(face.headEulerAngleY)
                                                           : [NSNull null],
              @"headEulerAngleZ" : face.hasHeadEulerAngleZ ? @(face.headEulerAngleZ)
                                                           : [NSNull null],
              @"smilingProbability" : smileProb,
              @"leftEyeOpenProbability" : leftProb,
              @"rightEyeOpenProbability" : rightProb,
              @"trackingId" : face.hasTrackingID ? @(face.trackingID) : [NSNull null],
              @"landmarks" : @{
                @"bottomMouth" : [FaceDetector getLandmarkPosition:face
                                                          landmark:FIRFaceLandmarkTypeMouthBottom],
                @"leftCheek" : [FaceDetector getLandmarkPosition:face
                                                        landmark:FIRFaceLandmarkTypeLeftCheek],
                @"leftEar" : [FaceDetector getLandmarkPosition:face
                                                      landmark:FIRFaceLandmarkTypeLeftEar],
                @"leftEye" : [FaceDetector getLandmarkPosition:face
                                                      landmark:FIRFaceLandmarkTypeLeftEye],
                @"leftMouth" : [FaceDetector getLandmarkPosition:face
                                                        landmark:FIRFaceLandmarkTypeMouthLeft],
                @"noseBase" : [FaceDetector getLandmarkPosition:face
                                                       landmark:FIRFaceLandmarkTypeNoseBase],
                @"rightCheek" : [FaceDetector getLandmarkPosition:face
                                                         landmark:FIRFaceLandmarkTypeRightCheek],
                @"rightEar" : [FaceDetector getLandmarkPosition:face
                                                       landmark:FIRFaceLandmarkTypeRightEar],
                @"rightEye" : [FaceDetector getLandmarkPosition:face
                                                       landmark:FIRFaceLandmarkTypeRightEye],
                @"rightMouth" : [FaceDetector getLandmarkPosition:face
                                                         landmark:FIRFaceLandmarkTypeMouthRight],
              },
              @"contours" : @{
                @"allPoints" : [FaceDetector getContourPoints:face contour:FIRFaceContourTypeAll],
                @"face" : [FaceDetector getContourPoints:face contour:FIRFaceContourTypeFace],
                @"leftEye" : [FaceDetector getContourPoints:face contour:FIRFaceContourTypeLeftEye],
                @"leftEyebrowBottom" :
                    [FaceDetector getContourPoints:face
                                           contour:FIRFaceContourTypeLeftEyebrowBottom],
                @"leftEyebrowTop" :
                    [FaceDetector getContourPoints:face contour:FIRFaceContourTypeLeftEyebrowTop],
                @"lowerLipBottom" :
                    [FaceDetector getContourPoints:face contour:FIRFaceContourTypeLowerLipBottom],
                @"lowerLipTop" : [FaceDetector getContourPoints:face
                                                        contour:FIRFaceContourTypeLowerLipTop],
                @"noseBottom" : [FaceDetector getContourPoints:face
                                                       contour:FIRFaceContourTypeNoseBottom],
                @"noseBridge" : [FaceDetector getContourPoints:face
                                                       contour:FIRFaceContourTypeNoseBridge],
                @"rightEye" : [FaceDetector getContourPoints:face
                                                     contour:FIRFaceContourTypeRightEye],
                @"rightEyebrowBottom" :
                    [FaceDetector getContourPoints:face
                                           contour:FIRFaceContourTypeRightEyebrowBottom],
                @"rightEyebrowTop" :
                    [FaceDetector getContourPoints:face contour:FIRFaceContourTypeRightEyebrowTop],
                @"upperLipBottom" :
                    [FaceDetector getContourPoints:face contour:FIRFaceContourTypeUpperLipBottom],
                @"upperLipTop" : [FaceDetector getContourPoints:face
                                                        contour:FIRFaceContourTypeUpperLipTop],
              }
            };

            [faceData addObject:data];
          }

          result(faceData);
        }];
}

+ (id)getLandmarkPosition:(MLKVisionFace *)face landmark:(FIRFaceLandmarkType)landmarkType {
    MLKFaceLandmark *landmark = [face landmarkOfType:landmarkType];
  if (landmark) {
    return @[ landmark.position.x, landmark.position.y ];
  }

  return [NSNull null];
}

+ (id)getContourPoints:(MLKVisionFace *)face contour:(FIRFaceContourType)contourType {
  MLKFaceContour *contour = [face contourOfType:contourType];
  if (contour) {
    NSArray<MLKVisionPoint *> *contourPoints = contour.points;
    NSMutableArray *result = [[NSMutableArray alloc] initWithCapacity:[contourPoints count]];
    for (int i = 0; i < [contourPoints count]; i++) {
      MLKVisionPoint *point = [contourPoints objectAtIndex:i];
      [result insertObject:@[ point.x, point.y ] atIndex:i];
    }
    return [result copy];
  }

  return [NSNull null];
}

+ (MLKFaceDetectorOptions *)parseOptions:(NSDictionary *)optionsData {
  MLKFaceDetectorOptions *options = [[MLKFaceDetectorOptions alloc] init];

  NSNumber *enableClassification = optionsData[@"enableClassification"];
  if (enableClassification.boolValue) {
    options.classificationMode = MLKFaceDetectorClassificationModeAll;
  } else {
    options.classificationMode = MLKFaceDetectorClassificationModeNone;
  }

  NSNumber *enableLandmarks = optionsData[@"enableLandmarks"];
  if (enableLandmarks.boolValue) {
    options.landmarkMode = MLKFaceDetectorLandmarkModeAll;
  } else {
    options.landmarkMode = MLKFaceDetectorLandmarkModeNone;
  }

  NSNumber *enableContours = optionsData[@"enableContours"];
  if (enableContours.boolValue) {
    options.contourMode = MLKFaceDetectorContourModeAll;
  } else {
    options.contourMode = MLKFaceDetectorContourModeNone;
  }

  NSNumber *enableTracking = optionsData[@"enableTracking"];
  options.trackingEnabled = enableTracking.boolValue;

  NSNumber *minFaceSize = optionsData[@"minFaceSize"];
  options.minFaceSize = [minFaceSize doubleValue];

  NSString *mode = optionsData[@"mode"];
  if ([mode isEqualToString:@"accurate"]) {
    options.performanceMode = MLKFaceDetectorPerformanceModeAccurate;
  } else if ([mode isEqualToString:@"fast"]) {
    options.performanceMode = MLKFaceDetectorPerformanceModeFast;
  }

  return options;
}
@end
