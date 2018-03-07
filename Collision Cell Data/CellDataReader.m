classdef CellDataReader < handle
%Opens a binary SRD output file and provides an interface for reading it frame by frame.
    
    properties (Access = private)
        fileID
		version
		format
		filesize

        obytes
        vbytes
		framesize
    end
    
    properties (GetAccess = public, SetAccess = private)
		cellCount
		ncells
        nframes
		collisionFreq
		samplingFreq
		outputFreq
		outputInterval
	end
    
    methods
        function obj = CellDataReader(filename)
            if exist(filename, 'file') ~= 2
                error('%s does not exist!', filename)
            end
            
            fileinfo = dir(filename);
            obj.filesize = fileinfo.bytes;
            
            obj.fileID = fopen(filename, 'r', 'b');
            magic = fread(obj.fileID, 1, 'uint32');
            
			% check endianness
            obj.format = 'b';
			if (magic ~= 2016) % 0x000007E0
				if(magic == 3758555136) % 0xE0070000
                    obj.format = 'l';
				else
                    error('%s magic number does not match!', filename)
				end
			end
			obj.version		= fread(obj.fileID, 1, 'uint32', 0, obj.format);
            obj.obytes		= fread(obj.fileID, 1, 'uint32', 0, obj.format);
            obj.vbytes		= fread(obj.fileID, 1, 'uint32', 0, obj.format);
			obj.cellCount	= fread(obj.fileID, 3, 'uint32', 0, obj.format)';
			obj.nframes		= fread(obj.fileID, 1, 'uint32', 0, obj.format)';
			obj.collisionFreq	= fread(obj.fileID, 1, 'uint32', 0, obj.format)';
			obj.samplingFreq	= fread(obj.fileID, 1, 'uint32', 0, obj.format)';
			obj.outputFreq		= fread(obj.fileID, 1, 'uint32', 0, obj.format)';
            obj.outputInterval	= fread(obj.fileID, 1, 'float', 0, obj.format)';
			
			obj.ncells = prod(obj.cellCount);
			obj.framesize = 32 + obj.obytes + obj.vbytes;
        end
        
        function b = hasOccupacyData(obj)
            b = (obj.obytes ~= 0);
        end
        
        function b = hasVelocityData(obj)
            b = (obj.vbytes ~= 0);
		end
        
        function occupancyData = getOccupacyData(obj, n)
            if(~hasOccupacyData(obj))
                error('file does not have occupancy data!');
            end
            
            fseek(obj.fileID, (obj.framesize * (n - 1)) + 80, 'bof');
            occupancyData = fread(obj.fileID, obj.ncells, 'float', 0, obj.format)';
        end
        
        function velocityData = getVelocityData(obj, n)
            if(~hasVelocityData(obj))
                error('file does not have velocity data!');
            end
            
            fseek(obj.fileID, (obj.framesize * (n - 1)) + 80 + obj.obytes, 'bof');
            velocityData = fread(obj.fileID, [3, obj.ncells], 'float', 0, obj.format)';
		end
		
		function frameTime = getFrameTime(obj, n)
            fseek(obj.fileID, (obj.framesize * (n - 1)) + 52, 'bof');
            frameTime = fread(obj.fileID, 1, 'float', 0, obj.format);
		end
		
		function gridShift = getGridShift(obj, n)
            fseek(obj.fileID, (obj.framesize * (n - 1)) + 56, 'bof');
            gridShift = fread(obj.fileID, 3, 'float', 0, obj.format);
        end
        
        function cellSize = getCellSize(obj, n)
            fseek(obj.fileID, (obj.framesize * (n - 1)) + 68, 'bof');
            cellSize = fread(obj.fileID, 3, 'float', 0, obj.format);
		end
		
		 function occupancyData = getAllOccupancyData(obj)
            occupancyData = zeros(obj.nframes, obj.ncells);
            for i = 1 : obj.nframes
                occupancyData(i,:) = getOccupacyData(obj, i);
            end
        end
        
        function velocityData = getAllVelocityData(obj)
            velocityData = zeros(obj.nframes, obj.ncells, 3);
            for i = 1 : obj.nframes
                velocityData(i,:,:) = getVelocityData(obj, i);
            end
		end
        
        function frameTimes = getAllFrameTimes(obj)
            frameTimes = zeros(obj.nframes, 1);
            for i = 1 : obj.nframes
                frameTimes(i,:) = getFrameTime(obj, i);
            end
		end
		
		function gridShifts = getGridShifts(obj)
			gridShifts = zeros(obj.nframes, 3);
            for i = 1 : obj.nframes
                gridShifts(i,:) = getGridShift(obj, i);
            end
		end
		
		function cellSizes = getAllCellSizes(obj)
			cellSizes = zeros(obj.nframes, 3);
            for i = 1 : obj.nframes
                cellSizes(i,:) = getCellSize(obj, i);
            end
		end
        
        function delete(obj)
            fclose(obj.fileID);
        end
	end
end
