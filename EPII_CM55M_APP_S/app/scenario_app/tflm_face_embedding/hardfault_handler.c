/*
 Copyright (c) 2020 Arm Limited (or its affiliates). All rights reserved.
 Use, modification and redistribution of this file is subject to your possession of a
 valid End User License Agreement for the Arm Product of which these examples are part of
 and your compliance with all applicable terms and conditions of such licence agreement.
 */

#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "WE2_device.h"

#if 0
/* HardFault handler implementation that prints a message
   then exits the program early.
 */
void HardFault_Handler(void)
{
#if 0
    printf("HardFault occurred!\n");
#endif
    for(;;) {}
}
#endif
void HardFault_Handler(void) {
	/* Handling SAU related secure faults */
	printf("\r\nEntering HardFault interrupt!\r\n");
	if (SAU->SFSR != 0) {
		if (SAU->SFSR & SAU_SFSR_INVEP_Msk) {
			/* Invalid Secure state entry point */
			printf(
					"SAU->SFSR:INVEP fault: Invalid entry point to secure world.\r\n");
		} else if (SAU->SFSR & SAU_SFSR_AUVIOL_Msk) {
			/* AUVIOL: SAU violation  */
			printf(
					"SAU->SFSR:AUVIOL fault: SAU violation. Access to secure memory from normal world.\r\n");
		} else if (SAU->SFSR & SAU_SFSR_INVTRAN_Msk) {
			/* INVTRAN: Invalid transition from secure to normal world  */
			printf(
					"SAU->SFSR:INVTRAN fault: Invalid transition from secure to normal world.\r\n");
		} else {
			printf("Another SAU error.\r\n");
		}
		if (SAU->SFSR & SAU_SFSR_SFARVALID_Msk) {
			/* SFARVALID: SFAR contain valid address that caused secure violation */
			printf("Address that caused SAU violation is 0x%X.\r\n", SAU->SFAR);
		}
	}

	/* Handling secure bus related faults */
	if (SCB->CFSR != 0) {
		if (SCB->CFSR & SCB_CFSR_IBUSERR_Msk) {
			/* IBUSERR: Instruction bus error on an instruction prefetch */
			printf(
					"SCB->BFSR:IBUSERR fault: Instruction bus error on an instruction prefetch.\r\n");
		} else if (SCB->CFSR & SCB_CFSR_PRECISERR_Msk) {
			/* PRECISERR: Instruction bus error on an instruction prefetch */
			printf("SCB->BFSR:PRECISERR fault: Precise data access error.\r\n");
		} else {
			printf("Security Another secure bus error 1.\r\n");
		}
		if (SCB->CFSR & SCB_CFSR_BFARVALID_Msk) {
			/* BFARVALID: BFAR contain valid address that caused secure violation */
			printf("Address that caused secure bus violation is 0x%X.\r\n",
					SCB->BFAR);
		}
	}

	/* Handling non-secure bus related faults */
	if (SCB_NS->CFSR != 0) {
		if (SCB_NS->CFSR & SCB_CFSR_IBUSERR_Msk) {
			/* IBUSERR: Instruction bus error on an instruction prefetch */
			printf(
					"SCB_NS->BFSR:IBUSERR fault: Instruction bus error on an instruction prefetch.\r\n");
		} else if (SCB_NS->CFSR & SCB_CFSR_PRECISERR_Msk) {
			/* PRECISERR: Data bus error on an data read/write */
			printf(
					"SCB_NS->BFSR:PRECISERR fault: Precise data access error.\r\n");
		} else {
			printf("Security Another secure bus error 2.\r\n");
		}
		if (SCB_NS->CFSR & SCB_CFSR_BFARVALID_Msk) {
			/* BFARVALID: BFAR contain valid address that caused secure violation */
			printf("Address that caused secure bus violation is 0x%X.\r\n",
					SCB_NS->BFAR);
		}
	}

#if 0
    /* Perform system RESET */
    SCB->AIRCR =
        (SCB->AIRCR & ~SCB_AIRCR_VECTKEY_Msk) | (0x05FAUL << SCB_AIRCR_VECTKEY_Pos) | SCB_AIRCR_SYSRESETREQ_Msk;
#else
	printf("SCB->CFSR:0x%08x\n", SCB->CFSR);
	printf("SCB->BFAR:0x%08x\n", SCB->BFAR);
	printf("SCB->HFSR:0x%08x\n", SCB->HFSR);
	for (;;) {
	}
#endif
}

void NMI_Handler(void) {
	printf("\r\nEntering NMI_Handler interrupt!\r\n");
	for (;;) {
	}
}

void MemManage_Handler(void) {
	printf("\r\nEntering MemManage_Handler interrupt!\r\n");
	for (;;) {
	}
}
void BusFault_Handler(void) {
	printf("\r\nEntering BusFault_Handler interrupt!\r\n");
	printf("SCB->CFSR:0x%08x\n", SCB->CFSR);
	printf("SCB->BFAR:0x%08x\n", SCB->BFAR);
	printf("SCB->HFSR:0x%08x\n", SCB->HFSR);
	for (;;) {
	}
}
void UsageFault_Handler(void) {
	printf("\r\n\r\n=== USAGE FAULT DIAGNOSIS ===\r\n");

	volatile uint32_t cfsr = SCB->CFSR;
	printf("SCB->CFSR: 0x%08X\r\n", (unsigned int)cfsr);

	/* UsageFault Status Register (UFSR) is bits [31:16] of CFSR */
	uint16_t ufsr = (cfsr >> 16) & 0xFFFF;
	if (ufsr > 0) {
		printf("[UsageFault Flags]:\r\n");
		if (ufsr & (1 << 9)) printf("  - DIVBYZERO: Divide by zero\r\n");
		if (ufsr & (1 << 8)) printf("  - UNALIGNED: Unaligned memory access\r\n");
		if (ufsr & (1 << 3)) printf("  - NOCP: Coprocessor not present or disabled\r\n");
		if (ufsr & (1 << 2)) printf("  - INVPC: Invalid PC load (illegal EXC_RETURN)\r\n");
		if (ufsr & (1 << 1)) printf("  - INVSTATE: Invalid EPSR.T or EPSR.IT\r\n");
		if (ufsr & (1 << 0)) printf("  - UNDEFINSTR: Undefined instruction\r\n");
	}

	/* BusFault Status Register (BFSR) is bits [15:8] of CFSR */
	uint8_t bfsr = (cfsr >> 8) & 0xFF;
	if (bfsr > 0) {
		printf("[BusFault Flags]:\r\n");
		if (bfsr & (1 << 7)) printf("  - BFARVALID: BFAR holds valid address\r\n");
		if (bfsr & (1 << 5)) printf("  - LSPERR: FP lazy state preservation error\r\n");
		if (bfsr & (1 << 4)) printf("  - STKERR: Stacking error\r\n");
		if (bfsr & (1 << 3)) printf("  - UNSTKERR: Unstacking error\r\n");
		if (bfsr & (1 << 2)) printf("  - IMPRECISERR: Imprecise data access error\r\n");
		if (bfsr & (1 << 1)) printf("  - PRECISERR: Precise data access error\r\n");
		if (bfsr & (1 << 0)) printf("  - IBUSERR: Instruction bus error\r\n");
		if (bfsr & (1 << 7)) {
			printf("  => Faulting Address (BFAR): 0x%08X\r\n", (unsigned int)SCB->BFAR);
		}
	}

	/* MemManage Fault Status Register (MMFSR) is bits [7:0] of CFSR */
	uint8_t mmfsr = cfsr & 0xFF;
	if (mmfsr > 0) {
		printf("[MemManage Fault Flags]:\r\n");
		if (mmfsr & (1 << 7)) printf("  - MMARVALID: MMFAR holds valid address\r\n");
		if (mmfsr & (1 << 5)) printf("  - MLSPERR: FP lazy state preservation error\r\n");
		if (mmfsr & (1 << 4)) printf("  - MSTKERR: Stacking error\r\n");
		if (mmfsr & (1 << 3)) printf("  - MUNSTKERR: Unstacking error\r\n");
		if (mmfsr & (1 << 1)) printf("  - DACCVIOL: Data access violation\r\n");
		if (mmfsr & (1 << 0)) printf("  - IACCVIOL: Instruction access violation\r\n");
		if (mmfsr & (1 << 7)) {
			printf("  => Faulting Address (MMFAR): 0x%08X\r\n", (unsigned int)SCB->MMFAR);
		}
	}

	/* HardFault Status Register */
	printf("SCB->HFSR: 0x%08X\r\n", (unsigned int)SCB->HFSR);
	if (SCB->HFSR & (1 << 30)) {
		printf("  - FORCED: Fault escalated to HardFault\r\n");
	}

	printf("=== END FAULT DIAGNOSIS ===\r\n\r\n");

	for (;;) {
	}
}
void SecureFault_Handler(void) {
	printf("\r\nEntering SecureFault_Handler interrupt!\r\n");
	for (;;) {
	}
}

